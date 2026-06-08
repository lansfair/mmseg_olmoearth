from __future__ import annotations

import argparse
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from tqdm.auto import tqdm

try:
    import rasterio
    from rasterio.warp import transform as transform_coords
    from rasterio.windows import Window
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "large_image_inference.py requires rasterio to read/write GeoTIFFs."
    ) from exc


S2_BANDS = [
    "B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12",
    "B01", "B09"
]
PACK_TYPES = {"PackSegInputs", "PackOlmoEarthSegInputs"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shared sliding-window large GeoTIFF inference for single-input "
            "MMSeg semantic segmentation configs."
        )
    )
    parser.add_argument("image", help="Input large GeoTIFF.")
    parser.add_argument("config", help="MMSeg semantic segmentation config.")
    parser.add_argument("checkpoint", help="MMSeg segmentation checkpoint.")
    parser.add_argument("output", help="Output label GeoTIFF.")
    parser.add_argument(
        "--input-mode",
        choices=["auto", "standard", "rgb", "s2", "copernicus"],
        default="auto",
        help=(
            "Input interpretation. auto selects Copernicus, OLMoEarth RGB, "
            "OLMoEarth S2, or standard mode from the config."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Sliding window size. Defaults to config crop_size/data_preprocessor size.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        help="Sliding stride. Defaults to the window size.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--band-indices",
        type=int,
        nargs="+",
        default=None,
        help="1-based input GeoTIFF bands to read in standard/Copernicus modes.",
    )
    parser.add_argument(
        "--band-scales",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-loaded-band scale factors in standard/Copernicus modes.",
    )
    parser.add_argument(
        "--rgb-channel-order",
        choices=["RGB", "BGR"],
        default=None,
        help="Band order for OLMoEarth RGB adapter GeoTIFF input.",
    )
    parser.add_argument(
        "--input-value-range",
        choices=["auto", "0_255", "0_1", "s2"],
        default=None,
        help="RGBToOlmoEarthS2 input value range override.",
    )
    parser.add_argument(
        "--source-band-names",
        nargs="+",
        default=None,
        help="Input S2 band names. Used to reorder bands to OLMoEarth order.",
    )
    parser.add_argument(
        "--s2-scale-factor",
        type=float,
        default=None,
        help="Optional scale factor applied to S2 windows before normalization.",
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        nargs=3,
        metavar=("DAY", "MONTH", "YEAR"),
        default=(1, 1, 2025),
        help="Timestamp metadata passed to OLMoEarth for each timestep.",
    )
    parser.add_argument(
        "--copernicus-date",
        default=None,
        help="Sensing date as YYYYMMDD. Overrides date parsed from filename.",
    )
    parser.add_argument(
        "--copernicus-date-days",
        type=float,
        default=None,
        help="Sensing time as days since 1970-01-01. Overrides date parsing.",
    )
    parser.add_argument(
        "--copernicus-lon-lat",
        type=float,
        nargs=2,
        metavar=("LON", "LAT"),
        default=None,
        help="Override geospatial lon/lat metadata for every window.",
    )
    parser.add_argument(
        "--copernicus-patch-area",
        type=float,
        default=None,
        help="Copernicus patch area metadata override.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, same syntax as tools/test.py.",
    )
    return parser.parse_args()


def get_pipeline(cfg: Config) -> list[dict[str, Any]]:
    pipeline = cfg.get("test_pipeline")
    if pipeline is None:
        pipeline = cfg.test_dataloader.dataset.pipeline
    return [dict(transform) for transform in pipeline]


def cfg_get_crop_size(cfg: Config) -> tuple[int, int]:
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
    for transform in pipeline:
        if transform.get("type") == transform_type:
            return transform
    return None


def clean_pipeline(
    pipeline: list[dict[str, Any]],
    extra_drop_types: set[str] | None = None,
) -> list[dict[str, Any]]:
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
    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg.custom_imports)


def build_model(cfg: Config, checkpoint: str, device: str):
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
    backbone = cfg.model.get("backbone", {})
    if not isinstance(backbone, dict):
        return {}
    return backbone


def backbone_type(cfg: Config) -> str:
    return str(backbone_cfg(cfg).get("type", ""))


def uses_olmoearth_feature_backbone(cfg: Config) -> bool:
    return backbone_type(cfg) == "OlmoEarthFeatureBackbone"


def validate_large_inference_config(cfg: Config, args: argparse.Namespace) -> None:
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
    src: rasterio.io.DatasetReader,
    grid: tuple[int, int, int, int, int, int, int, int],
    band_indices: list[int] | None = None,
    band_scales: list[float] | None = None,
    nan_to_num: bool = False,
    to_float32: bool = False,
) -> np.ndarray:
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
    sample = SegDataSample()
    sample.set_metainfo(metainfo)
    return sample


def pack_window(
    results: dict[str, Any],
    pipeline: Compose,
    meta_keys: tuple[str, ...],
) -> dict[str, Any]:
    results = pipeline(results)
    img = results["img"]
    if img.ndim < 3:
        img = np.expand_dims(img, -1)
    inputs = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1))).contiguous()
    metainfo = {key: results[key] for key in meta_keys if key in results}
    return {"inputs": inputs, "data_samples": data_sample(metainfo)}


def predict_batch(model, batch_items: list[dict[str, Any]]) -> list[np.ndarray]:
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
    if cfg.get("num_classes") is not None:
        return int(cfg.num_classes)
    decode_head = cfg.model.get("decode_head", {})
    if isinstance(decode_head, dict) and decode_head.get("num_classes") is not None:
        return int(decode_head["num_classes"])
    return None


def output_dtype(class_count: int | None) -> str:
    if class_count is not None and class_count <= 255:
        return "uint8"
    if class_count is not None and class_count <= 32767:
        return "int16"
    return "int32"


def normalize_band_name(name: str) -> str:
    return str(name).strip().upper()


def band_indices_for_s2(
    source_band_names: list[str] | None,
    raster_count: int,
) -> list[int]:
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
    if value is None:
        return None
    return [float(item) for item in value]


def as_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in value]


def parse_days_from_date(date_text: str | None) -> float:
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
    if separator is None:
        return float("nan")
    basename = Path(filename).name
    try:
        token = basename.split(separator)[int(token_index)]
    except IndexError:
        return float("nan")
    return parse_days_from_date(token)


def window_lon_lat(
    src: rasterio.io.DatasetReader,
    grid: tuple[int, int, int, int, int, int, int, int],
    lon_lat_override: tuple[float, float] | None,
) -> tuple[float, float]:
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


class BaseAdapter:
    name = "standard"
    meta_keys = (
        "img_path",
        "ori_shape",
        "img_shape",
        "pad_shape",
        "scale_factor",
        "flip",
        "flip_direction",
        "reduce_zero_label",
    )

    def __init__(
        self,
        cfg: Config,
        raw_pipeline: list[dict[str, Any]],
        args: argparse.Namespace,
    ) -> None:
        self.cfg = cfg
        self.raw_pipeline = raw_pipeline
        self.args = args
        self.pipeline = Compose(clean_pipeline(raw_pipeline))
        self.band_indices = args.band_indices
        self.band_scales = args.band_scales
        self.nan_to_num = False
        self.to_float32 = False

    @classmethod
    def detect(cls, cfg: Config, raw_pipeline: list[dict[str, Any]]) -> bool:
        return True

    def prepare(self, src: rasterio.io.DatasetReader) -> None:
        if self.args.input_mode == "rgb" and self.band_indices is None:
            self.band_indices = [1, 2, 3]

    def read(
        self,
        src: rasterio.io.DatasetReader,
        grid: tuple[int, int, int, int, int, int, int, int],
    ) -> np.ndarray:
        return read_window(
            src,
            grid,
            self.band_indices,
            self.band_scales,
            self.nan_to_num,
            self.to_float32,
        )

    def make_results(
        self,
        image: np.ndarray,
        src: rasterio.io.DatasetReader,
        grid: tuple[int, int, int, int, int, int, int, int],
    ) -> dict[str, Any]:
        return {
            "img": image,
            "img_path": self.args.image,
            "img_shape": image.shape[:2],
            "ori_shape": image.shape[:2],
        }

    def pack(
        self,
        image: np.ndarray,
        src: rasterio.io.DatasetReader,
        grid: tuple[int, int, int, int, int, int, int, int],
    ) -> dict[str, Any]:
        return pack_window(
            self.make_results(image, src, grid),
            self.pipeline,
            self.meta_keys,
        )


class OlmoEarthAdapter(BaseAdapter):
    name = "olmoearth"
    meta_keys = BaseAdapter.meta_keys + (
        "olmoearth_modality",
        "olmoearth_num_timesteps",
        "olmoearth_band_names",
        "present_bands",
        "timestamps",
        "olmoearth_rgb_adapter",
        "olmoearth_raw_band_names",
    )

    @classmethod
    def detect(cls, cfg: Config, raw_pipeline: list[dict[str, Any]]) -> bool:
        return (
            find_transform(raw_pipeline, "RGBToOlmoEarthS2") is not None
            or find_transform(raw_pipeline, "OlmoEarthNormalize") is not None
            or "OlmoEarth" in str(cfg.model.get("type", ""))
            or "OlmoEarth" in str(cfg.model.get("backbone", {}).get("type", ""))
        )

    def __init__(
        self,
        cfg: Config,
        raw_pipeline: list[dict[str, Any]],
        args: argparse.Namespace,
    ) -> None:
        super().__init__(cfg, raw_pipeline, args)
        import_modules_from_strings(
            imports=["projects.olmoearth.olmoearth"],
            allow_failed_imports=False,
        )
        transforms = clean_pipeline(raw_pipeline)
        self.olmo_mode = self._resolve_olmo_mode()
        if self.olmo_mode == "rgb":
            rgb = find_transform(transforms, "RGBToOlmoEarthS2")
            if rgb is None:
                rgb = dict(type="RGBToOlmoEarthS2")
                transforms.append(rgb)
            rgb["rgb_channel_order"] = args.rgb_channel_order or "RGB"
            if args.input_value_range is not None:
                rgb["input_value_range"] = args.input_value_range
            else:
                rgb.setdefault("input_value_range", "auto")
            self.band_indices = [1, 2, 3]
        elif self.olmo_mode == "s2":
            if find_transform(transforms, "OlmoEarthNormalize") is None:
                transforms.append(
                    dict(
                        type="OlmoEarthNormalize",
                        modality="sentinel2_l2a",
                        num_timesteps=cfg.get("num_timesteps", 1),
                    )
                )
            self.to_float32 = args.s2_scale_factor is not None
        self.pipeline = Compose(transforms)

    def _resolve_olmo_mode(self) -> str:
        if self.args.input_mode in {"rgb", "s2"}:
            return self.args.input_mode
        if find_transform(self.raw_pipeline, "RGBToOlmoEarthS2") is not None:
            return "rgb"
        return "s2"

    def prepare(self, src: rasterio.io.DatasetReader) -> None:
        if self.olmo_mode == "s2":
            self.band_indices = band_indices_for_s2(
                self.args.source_band_names,
                src.count,
            )
        elif src.count < 3:
            raise ValueError(f"RGB mode requires at least 3 bands, got {src.count}")

    def read(
        self,
        src: rasterio.io.DatasetReader,
        grid: tuple[int, int, int, int, int, int, int, int],
    ) -> np.ndarray:
        image = super().read(src, grid)
        if self.olmo_mode == "s2" and self.args.s2_scale_factor is not None:
            image = image.astype(np.float32, copy=False) * self.args.s2_scale_factor
        return image

    def make_results(
        self,
        image: np.ndarray,
        src: rasterio.io.DatasetReader,
        grid: tuple[int, int, int, int, int, int, int, int],
    ) -> dict[str, Any]:
        results = super().make_results(image, src, grid)
        results["timestamps"] = np.asarray([tuple(self.args.timestamp)], dtype=np.int64)
        if self.olmo_mode == "s2":
            results["olmoearth_modality"] = "sentinel2_l2a"
            results["olmoearth_num_timesteps"] = 1
            results["olmoearth_band_names"] = S2_BANDS
            results["present_bands"] = S2_BANDS
        return results


class CopernicusAdapter(BaseAdapter):
    name = "copernicus"
    meta_keys = BaseAdapter.meta_keys + ("copernicus_meta",)

    @classmethod
    def detect(cls, cfg: Config, raw_pipeline: list[dict[str, Any]]) -> bool:
        model_type = str(cfg.model.get("type", ""))
        backbone_type = str(cfg.model.get("backbone", {}).get("type", ""))
        transform_types = {str(transform.get("type")) for transform in raw_pipeline}
        return (
            "Copernicus" in model_type
            or "Copernicus" in backbone_type
            or "LoadCopernicusGeoTiffImageFromFile" in transform_types
            or "AddCopernicusMeta" in transform_types
        )

    def __init__(
        self,
        cfg: Config,
        raw_pipeline: list[dict[str, Any]],
        args: argparse.Namespace,
    ) -> None:
        super().__init__(cfg, raw_pipeline, args)
        self.pipeline = Compose(clean_pipeline(raw_pipeline, {"AddCopernicusMeta"}))
        self.loader = find_transform(raw_pipeline, "LoadCopernicusGeoTiffImageFromFile")
        self.to_float32 = True
        self.nan_to_num = True
        if self.loader is not None:
            self.band_indices = self.band_indices or as_int_list(
                self.loader.get("band_indices")
            )
            self.band_scales = self.band_scales or as_float_list(
                self.loader.get("band_scales")
            )
            self.nan_to_num = bool(self.loader.get("nan_to_num", True))
            self.to_float32 = bool(self.loader.get("to_float32", True))
        self.patch_area = self._get_patch_area()
        self.sensing_time = self._get_sensing_time()
        self.lon_lat_override = (
            tuple(args.copernicus_lon_lat)
            if args.copernicus_lon_lat is not None
            else None
        )

    def _get_patch_area(self) -> float:
        if self.args.copernicus_patch_area is not None:
            return float(self.args.copernicus_patch_area)
        if self.loader is not None and self.loader.get("patch_area") is not None:
            return float(self.loader["patch_area"])
        add_meta = find_transform(self.raw_pipeline, "AddCopernicusMeta")
        if add_meta is not None and add_meta.get("patch_area") is not None:
            return float(add_meta["patch_area"])
        patch_area = self.cfg.model.get("backbone", {}).get("patch_area")
        if patch_area is not None:
            return float(patch_area)
        return float("nan")

    def _get_sensing_time(self) -> float:
        if self.args.copernicus_date_days is not None:
            return float(self.args.copernicus_date_days)
        parsed = parse_days_from_date(self.args.copernicus_date)
        if not np.isnan(parsed):
            return parsed
        if self.loader is None:
            return float("nan")
        return parse_days_from_filename(
            self.args.image,
            self.loader.get("date_separator"),
            int(self.loader.get("date_token_index", 1)),
        )

    def make_results(
        self,
        image: np.ndarray,
        src: rasterio.io.DatasetReader,
        grid: tuple[int, int, int, int, int, int, int, int],
    ) -> dict[str, Any]:
        results = super().make_results(image, src, grid)
        lon, lat = window_lon_lat(src, grid, self.lon_lat_override)
        results["copernicus_meta"] = np.array(
            [lon, lat, self.sensing_time, self.patch_area],
            dtype=np.float32,
        )
        return results


def select_adapter(
    cfg: Config,
    raw_pipeline: list[dict[str, Any]],
    args: argparse.Namespace,
) -> BaseAdapter:
    if uses_olmoearth_feature_backbone(cfg):
        if args.input_mode in {"rgb", "s2", "copernicus"}:
            raise ValueError(
                "OlmoEarthFeatureBackbone configs are offline embedding "
                "configs. They cannot run raw image adapters "
                f"(--input-mode {args.input_mode})."
            )
        return BaseAdapter(cfg, raw_pipeline, args)
    if args.input_mode == "copernicus":
        return CopernicusAdapter(cfg, raw_pipeline, args)
    if args.input_mode in {"rgb", "s2"} and OlmoEarthAdapter.detect(cfg, raw_pipeline):
        return OlmoEarthAdapter(cfg, raw_pipeline, args)
    if args.input_mode == "auto":
        if CopernicusAdapter.detect(cfg, raw_pipeline):
            return CopernicusAdapter(cfg, raw_pipeline, args)
        if OlmoEarthAdapter.detect(cfg, raw_pipeline):
            return OlmoEarthAdapter(cfg, raw_pipeline, args)
    return BaseAdapter(cfg, raw_pipeline, args)


def run_large_inference(args: argparse.Namespace) -> None:
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


def main() -> None:
    run_large_inference(parse_args())
