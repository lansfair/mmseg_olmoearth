from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

from ..utils import get_modality_bands

try:
    from osgeo import gdal
except ImportError:
    gdal = None


def _load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError(
                "Reading GeoTIFF inputs requires rasterio."
            ) from exc
        with rasterio.open(path) as src:
            array = src.read()
        if array.shape[0] == 1:
            return array[0]
        return array
    raise ValueError(f"Only GeoTIFF arrays are supported, got: {path}")


DFC2020_S2_ALL_BANDS = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
)


def _tchw_to_hw_flat(image: np.ndarray) -> np.ndarray:
    if image.ndim != 4:
        raise ValueError(f"Expected stacked GeoTIFF image as TCHW, got {image.shape}")
    t, c, h, w = image.shape
    return image.transpose(2, 3, 1, 0).reshape(h, w, c * t)


def _load_multitif(paths: list[str | Path]) -> np.ndarray:
    images = []
    for path in paths:
        image = _load_array(path)
        if image.ndim == 2:
            image = image[None, ...]
        if image.ndim != 3:
            raise ValueError(
                "Each path in img_paths must load to CHW or HW GeoTIFF data, "
                f"got shape {image.shape} from {path}"
            )
        images.append(image)
    shape_set = {image.shape for image in images}
    if len(shape_set) != 1:
        raise ValueError(
            "All img_paths in one sample must have the same CHW shape, "
            f"got {sorted(shape_set)}"
        )
    return np.stack(images, axis=0)


@TRANSFORMS.register_module()
class LoadOlmoEarthArrays(BaseTransform):
    """Load OLMoEarth GeoTIFF images, labels, optional masks and timestamps."""

    def __init__(
        self,
        ignore_index: int = 255,
        source_ignore_values: tuple[int, ...] = (-1,),
        reduce_zero_label: bool = False,
    ) -> None:
        self.ignore_index = ignore_index
        self.source_ignore_values = source_ignore_values
        self.reduce_zero_label = reduce_zero_label

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        results["seg_fields"] = ["gt_seg_map"]
        if "img_paths" not in results:
            raise KeyError(
                "Manifest samples must provide 'img_paths' with one GeoTIFF "
                "per timestep."
            )
        image = _load_multitif(results["img_paths"]).astype(
            np.float32,
            copy=False,
        )
        results["img"] = _tchw_to_hw_flat(image)
        results["img_shape"] = results["img"].shape[:2]
        results["ori_shape"] = results["img"].shape[:2]

        label = _load_array(results["seg_map_path"]).squeeze().astype(np.int64)
        if self.reduce_zero_label:
            label = label.copy()
            label[label == 0] = self.ignore_index
            label = label - 1
            label[label == self.ignore_index - 1] = self.ignore_index
        if self.source_ignore_values:
            label = label.copy()
            for value in self.source_ignore_values:
                label[label == value] = self.ignore_index
        results["gt_seg_map"] = label

        valid_mask_path = results.get("valid_mask_path")
        if valid_mask_path is not None:
            valid = _load_array(valid_mask_path).squeeze().astype(np.float32)
            results["gt_valid_mask"] = valid
            results["seg_fields"].append("gt_valid_mask")

        if "timestamps" in results:
            results["timestamps"] = np.asarray(
                results["timestamps"],
                dtype=np.int64,
            )

        return results


@TRANSFORMS.register_module()
class LoadOlmoEarthDFC2020S2Image(BaseTransform):
    """Load Copernicus-Bench DFC2020-S2 GeoTIFFs for OLMoEarth.

    Copernicus-Bench stores 13 Sentinel-2 bands. OLMoEarth's Sentinel-2 L2A
    interface uses 12 bands and omits B10, so this transform selects and
    reorders bands from ``img_path`` into OLMoEarth order.
    """

    def __init__(
        self,
        band_names: list[str] | None = None,
        all_band_names: tuple[str, ...] = DFC2020_S2_ALL_BANDS,
        default_timestamp: tuple[int, int, int] = (15, 6, 2020),
        scale_factor: float | None = None,
        clip_range: tuple[float, float] | None = None,
    ) -> None:
        self.band_names = band_names or list(get_modality_bands("sentinel2_l2a"))
        self.all_band_names = tuple(all_band_names)
        self.band_indices = [
            self.all_band_names.index(band_name) + 1
            for band_name in self.band_names
        ]
        self.default_timestamp = tuple(int(x) for x in default_timestamp)
        self.scale_factor = scale_factor
        self.clip_range = clip_range

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError(
                "LoadOlmoEarthDFC2020S2Image requires rasterio to read GeoTIFFs."
            ) from exc

        with rasterio.open(results["img_path"]) as src:
            image = src.read(self.band_indices).astype(np.float32)
        if self.scale_factor is not None:
            image *= self.scale_factor
        if self.clip_range is not None:
            image = np.clip(image, self.clip_range[0], self.clip_range[1])
        results["img"] = np.ascontiguousarray(image.transpose(1, 2, 0))
        results["img_shape"] = results["img"].shape[:2]
        results["ori_shape"] = results["img"].shape[:2]
        results["olmoearth_modality"] = "sentinel2_l2a"
        results["olmoearth_num_timesteps"] = 1
        results["olmoearth_band_names"] = list(self.band_names)
        results["present_bands"] = list(self.band_names)
        results["timestamps"] = np.asarray([self.default_timestamp], dtype=np.int64)
        return results


@TRANSFORMS.register_module()
class LoadDFC2020Annotations(BaseTransform):
    """Load and remap original DFC2020 labels to 8 valid classes."""

    cls_mapping = {
        0: 255,
        1: 0,
        2: 1,
        3: 255,
        4: 2,
        5: 3,
        6: 4,
        7: 5,
        8: 255,
        9: 6,
        10: 7,
    }

    def __init__(self):
        if gdal is None:
            raise RuntimeError("gdal is not installed")

    def transform(self, results):
        ds = gdal.Open(results["seg_map_path"])
        if ds is None:
            raise FileNotFoundError(
                f'Unable to open file: {results["seg_map_path"]}'
            )
        seg_map = ds.ReadAsArray()
        remapped = np.full(seg_map.shape, 255, dtype=np.uint8)
        for old_label, new_label in self.cls_mapping.items():
            remapped[seg_map == old_label] = new_label
        results["gt_seg_map"] = remapped
        results.setdefault("seg_fields", []).append("gt_seg_map")
        return results
