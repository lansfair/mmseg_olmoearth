from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

S2_BAND_MEAN = np.array(
    [
        1711.0938,
        1308.8511,
        1546.4543,
        3010.1293,
        3106.5083,
        2068.3044,
        2685.0845,
        2931.5889,
        2514.6928,
        1899.4922,
    ],
    dtype=np.float32,
)
S2_BAND_STD = np.array(
    [
        1926.1026,
        1862.9751,
        1803.1792,
        1741.7837,
        1677.4543,
        1888.7862,
        1736.3090,
        1715.8104,
        1514.5199,
        1398.4779,
    ],
    dtype=np.float32,
)
S1_BAND_MEAN = np.array([5484.0407, 3003.7812], dtype=np.float32)
S1_BAND_STD = np.array([1871.2334, 1726.0670], dtype=np.float32)


def _read_array(path: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix in {".tif", ".tiff"}:
        import rasterio

        with rasterio.open(path) as src:
            return src.read()
    raise ValueError(f"Unsupported array format for {path}")


def _read_label(path: str) -> np.ndarray:
    label = _read_array(path)
    if label.ndim == 3:
        if label.shape[0] == 1:
            label = label[0]
        elif label.shape[-1] == 1:
            label = label[..., 0]
        else:
            raise ValueError(
                f"Expected a single-channel label map, got {label.shape}"
            )
    return label.astype(np.int64, copy=False)


def _choose_indices(
    valid_idx: np.ndarray,
    target: int,
    rng: np.random.Generator,
    random_sample: bool,
) -> np.ndarray:
    if len(valid_idx) == 0:
        return np.array([], dtype=np.int64)
    replace = len(valid_idx) < target
    if random_sample:
        chosen = rng.choice(valid_idx, size=target, replace=replace)
        return np.sort(chosen)
    if replace:
        positions = np.linspace(0, len(valid_idx) - 1, num=target)
        return valid_idx[np.rint(positions).astype(np.int64)]
    chunks = np.array_split(valid_idx, target)
    return np.array([chunk[len(chunk) // 2] for chunk in chunks], dtype=np.int64)


@TRANSFORMS.register_module()
class LoadTesseraTemporalArrays(BaseTransform):
    """Load standard TESSERA preprocessed tile arrays for online MMSeg use.

    The transform converts variable-length annual S2/S1 observations into a
    fixed per-pixel tensor compatible with ``TesseraBackbone``:
    ``H x W x (sample_size_s2 * 11 + sample_size_s1 * 3)``.
    """

    def __init__(
        self,
        sample_size_s2: int = 40,
        sample_size_s1: int = 40,
        random_sample: bool = True,
        standardize: bool = True,
        ignore_index: int = 255,
    ) -> None:
        self.sample_size_s2 = int(sample_size_s2)
        self.sample_size_s1 = int(sample_size_s1)
        self.random_sample = bool(random_sample)
        self.standardize = bool(standardize)
        self.ignore_index = int(ignore_index)

    def _resolve_tile_file(self, results: dict[str, Any], name: str) -> str:
        key = f"{name}_path"
        if key in results:
            return results[key]
        tile_path = results.get("tile_path")
        if tile_path is None:
            raise KeyError(
                "LoadTesseraTemporalArrays requires either tile_path or "
                f"{key}."
            )
        return str(Path(tile_path) / f"{name}.npy")

    def _sample_s2_pixel(
        self,
        bands: np.ndarray,
        masks: np.ndarray,
        doys: np.ndarray,
        i: int,
        j: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        pixel_bands = bands[:, i, j, :].astype(np.float32, copy=False)
        valid_idx = np.nonzero(masks[:, i, j])[0]
        chosen = _choose_indices(
            valid_idx,
            self.sample_size_s2,
            rng,
            self.random_sample,
        )
        if len(chosen) == 0:
            return np.zeros((self.sample_size_s2, 11), dtype=np.float32)
        sub_bands = pixel_bands[chosen]
        if self.standardize:
            sub_bands = (sub_bands - S2_BAND_MEAN) / (S2_BAND_STD + 1e-9)
        sub_doys = doys[chosen].astype(np.float32, copy=False)
        return np.hstack([sub_bands, sub_doys[:, None]]).astype(np.float32)

    def _sample_s1_pixel(
        self,
        asc_bands: np.ndarray,
        asc_doys: np.ndarray,
        desc_bands: np.ndarray,
        desc_doys: np.ndarray,
        i: int,
        j: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        parts_bands = []
        parts_doys = []
        for bands, doys in ((asc_bands, asc_doys), (desc_bands, desc_doys)):
            if bands.shape[0] == 0:
                continue
            pixel_bands = bands[:, i, j, :].astype(np.float32, copy=False)
            valid_idx = np.nonzero(np.any(pixel_bands != 0, axis=-1))[0]
            if len(valid_idx) == 0:
                continue
            parts_bands.append(pixel_bands[valid_idx])
            parts_doys.append(doys[valid_idx].astype(np.float32, copy=False))
        if not parts_bands:
            return np.zeros((self.sample_size_s1, 3), dtype=np.float32)

        all_bands = np.concatenate(parts_bands, axis=0)
        all_doys = np.concatenate(parts_doys, axis=0)
        valid_idx = np.arange(all_bands.shape[0], dtype=np.int64)
        chosen = _choose_indices(
            valid_idx,
            self.sample_size_s1,
            rng,
            self.random_sample,
        )
        sub_bands = all_bands[chosen]
        if self.standardize:
            sub_bands = (sub_bands - S1_BAND_MEAN) / (S1_BAND_STD + 1e-9)
        return np.hstack([sub_bands, all_doys[chosen, None]]).astype(np.float32)

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        bands = np.load(self._resolve_tile_file(results, "bands"), mmap_mode="r")
        masks = np.load(self._resolve_tile_file(results, "masks"), mmap_mode="r")
        doys = np.load(self._resolve_tile_file(results, "doys"), mmap_mode="r")
        asc_bands = np.load(
            self._resolve_tile_file(results, "sar_ascending"),
            mmap_mode="r",
        )
        asc_doys = np.load(
            self._resolve_tile_file(results, "sar_ascending_doy"),
            mmap_mode="r",
        )
        desc_bands = np.load(
            self._resolve_tile_file(results, "sar_descending"),
            mmap_mode="r",
        )
        desc_doys = np.load(
            self._resolve_tile_file(results, "sar_descending_doy"),
            mmap_mode="r",
        )

        _, height, width, _ = bands.shape
        channels = self.sample_size_s2 * 11 + self.sample_size_s1 * 3
        image = np.zeros((height, width, channels), dtype=np.float32)
        rng = np.random.default_rng()

        for i in range(height):
            for j in range(width):
                s2 = self._sample_s2_pixel(bands, masks, doys, i, j, rng)
                s1 = self._sample_s1_pixel(
                    asc_bands,
                    asc_doys,
                    desc_bands,
                    desc_doys,
                    i,
                    j,
                    rng,
                )
                image[i, j, :] = np.concatenate(
                    [s2.reshape(-1), s1.reshape(-1)],
                    axis=0,
                )

        label = _read_label(results["seg_map_path"])
        results["img"] = image
        results["gt_seg_map"] = label
        results["ori_shape"] = label.shape
        results["img_shape"] = image.shape[:2]
        valid_mask_path = results.get("valid_mask_path")
        if valid_mask_path:
            valid_mask = _read_array(valid_mask_path)
            if valid_mask.ndim == 3:
                valid_mask = np.squeeze(valid_mask)
            results["gt_valid_mask"] = valid_mask.astype(np.float32, copy=False)
        return results
