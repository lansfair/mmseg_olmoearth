"""Loading and packing transforms for UniverSat multimodal inputs."""

from typing import Dict, Optional

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.structures import PixelData
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


def _load_array(path: str) -> np.ndarray:
    """Load a numpy array from ``path``.

    Supports ``.npy`` files. Extend this helper if your modalities are stored
    in a different format (e.g. GeoTIFF via rasterio/gdal).
    """
    if path.endswith(".npy"):
        return np.load(path)
    raise ValueError(
        f"Unsupported modality file format for {path}. "
        f"UniverSat transforms currently support .npy files."
    )


@TRANSFORMS.register_module()
class LoadMultimodalFromFile(BaseTransform):
    """Load each modality raster from ``modality_paths``.

    Required keys:
        - ``modality_paths`` (dict): modality_name -> file path.

    Added keys:
        - ``img`` (dict): modality and optional ``<modality>_dates`` tensors.
        - ``img_shape`` / ``ori_shape``: shape of the first modality.

    A spatial snapshot must have shape ``(C, H, W)``. A time series must have
    shape ``(T, C, H, W)`` and provide relative day indices either through
    ``modality_dates`` or ``modality_date_paths``. Keeping snapshots three
    dimensional is important: the UniverSat encoder itself inserts their
    singleton time dimension.
    """

    def __init__(self, modalities: Optional[list] = None):
        self.modalities = modalities

    def transform(self, results: dict) -> dict:
        modality_paths: Dict[str, str] = results["modality_paths"]
        modalities = self.modalities or list(modality_paths.keys())

        date_paths = results.get("modality_date_paths", {})
        inline_dates = results.get("modality_dates", {})
        img = {}
        for mod in modalities:
            if mod not in modality_paths:
                raise KeyError(
                    f"Modality {mod!r} not found in modality_paths. "
                    f"Available: {list(modality_paths.keys())}"
                )
            array = _load_array(modality_paths[mod]).astype(np.float32)
            tensor = torch.from_numpy(array)
            if tensor.ndim not in (3, 4):
                raise ValueError(
                    f"Modality {mod!r} must have shape (C,H,W) or "
                    f"(T,C,H,W), got {tuple(tensor.shape)}."
                )
            img[mod] = tensor

            dates = inline_dates.get(mod)
            if mod in date_paths:
                if dates is not None:
                    raise ValueError(
                        f"Dates for {mod!r} were provided both inline and as a file."
                    )
                dates = _load_array(date_paths[mod])

            if tensor.ndim == 4:
                if dates is None:
                    raise ValueError(
                        f"Time-series modality {mod!r} requires relative day "
                        "indices in `dates` or `date_filenames`."
                    )
                dates = torch.as_tensor(dates, dtype=torch.long).flatten()
                if dates.numel() != tensor.shape[0]:
                    raise ValueError(
                        f"Modality {mod!r} has {tensor.shape[0]} time steps but "
                        f"{dates.numel()} dates."
                    )
                img[f"{mod}_dates"] = dates
            elif dates is not None:
                raise ValueError(
                    f"Spatial-only modality {mod!r} must not define dates."
                )

        results["img"] = img
        # Use the first modality's spatial shape as the reference shape.
        first = next(iter(img.values()))
        results["img_shape"] = tuple(first.shape[-2:])
        results["ori_shape"] = tuple(first.shape[-2:])
        return results


@TRANSFORMS.register_module()
class NormalizeMultimodal(BaseTransform):
    """Normalize each modality independently.

    Args:
        mean: dict mapping modality_name -> list of per-channel means.
        std: dict mapping modality_name -> list of per-channel stds.
    """

    def __init__(
        self,
        mean: Dict[str, list],
        std: Dict[str, list],
    ):
        self.mean = {
            mod: torch.tensor(vals, dtype=torch.float32).view(-1, 1, 1)
            for mod, vals in mean.items()
        }
        self.std = {
            mod: torch.tensor(vals, dtype=torch.float32).view(-1, 1, 1)
            for mod, vals in std.items()
        }

    def transform(self, results: dict) -> dict:
        img = results["img"]
        for mod, tensor in img.items():
            if mod.endswith("_dates"):
                continue
            mean = self.mean.get(mod)
            std = self.std.get(mod)
            if mean is None or std is None:
                continue
            # tensor shape is (T, C, H, W) or (C, H, W).
            if tensor.ndim == 4:
                m = mean.view(1, -1, 1, 1)
                s = std.view(1, -1, 1, 1)
            else:
                m = mean
                s = std
            img[mod] = (tensor - m) / s.clamp_min(1e-6)
        return results


@TRANSFORMS.register_module()
class PackUniverSatInputs(BaseTransform):
    """Pack multimodal ``img`` dict and seg map into MMSegmentation inputs.

    Required keys:
        - ``img`` (dict of tensors)
        - ``gt_seg_map`` (np.ndarray or tensor), optional

    Added keys:
        - ``inputs`` (dict of tensors)
        - ``data_samples`` (SegDataSample)
    """

    def __init__(
        self,
        meta_keys: tuple = (
            "img_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
        ),
    ):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed = {}
        packed["inputs"] = results["img"]

        data_sample = SegDataSample()
        if "gt_seg_map" in results:
            gt_seg_map = torch.as_tensor(results["gt_seg_map"]).long()
            if gt_seg_map.ndim == 2:
                gt_seg_map = gt_seg_map.unsqueeze(0)
            if gt_seg_map.ndim != 3 or gt_seg_map.shape[0] != 1:
                raise ValueError(
                    "gt_seg_map must have shape (H,W) or (1,H,W), got "
                    f"{tuple(gt_seg_map.shape)}."
                )
            data_sample.gt_sem_seg = PixelData(data=gt_seg_map)

        # Attach useful metadata.
        img_meta = {key: results[key] for key in self.meta_keys if key in results}
        data_sample.set_metainfo(img_meta)
        packed["data_samples"] = data_sample
        return packed
