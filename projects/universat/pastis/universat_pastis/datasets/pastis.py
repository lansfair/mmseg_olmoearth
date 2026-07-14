"""PASTIS-R dataset for UniverSat downstream evaluation.

This follows the original ``UniverSat/src/data/Pastis.py`` logic as closely as
possible while fitting into the MMSegmentation 1.x data pipeline.

Expected directory layout::

    data_root/
      metadata.geojson          # must contain ID_PATCH, Fold, dates-S2, dates-S1A
      DATA_S2/S2_{id}.npy       # T x 10 x H x W
      DATA_S1A/S1A_{id}.npy     # T x 3 x H x W
      ANNOTATIONS/TARGET_{id}.npy  # 1 x H x W or H x W
      NORM_S2_patch.json        # {"mean": [...], "std": [...]}
      NORM_S1_patch.json        # {"mean": [...], "std": [...]}
"""

import ast
import copy
import json
import os
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from mmengine.dataset import BaseDataset
from mmengine.registry import FUNCTIONS
from mmseg.registry import DATASETS

from ..utils.norm import apply_norm, load_norm


PASTIS_CLASSES = (
    "background",
    "class_1",
    "class_2",
    "class_3",
    "class_4",
    "class_5",
    "class_6",
    "class_7",
    "class_8",
    "class_9",
    "class_10",
    "class_11",
    "class_12",
    "class_13",
    "class_14",
    "class_15",
    "class_16",
    "class_17",
    "class_18",
)

PASTIS_PALETTE = [
    [0, 0, 0],
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 190],
    [0, 128, 128],
    [230, 190, 255],
    [170, 110, 40],
    [255, 250, 200],
    [128, 0, 0],
    [170, 255, 195],
    [128, 128, 0],
    [255, 215, 180],
]


def _prepare_dates(date_dict, reference_date: datetime):
    """Convert date strings/JSON/dict to relative day indices.

    PASTIS-R stores acquisition dates in ``metadata.geojson`` either as a
    JSON-encoded list ``["20190414", ...]`` or as a dict
    ``{"0": "20190414", ...}``. Both forms are accepted.
    """
    if isinstance(date_dict, str):
        try:
            date_dict = json.loads(date_dict)
        except json.JSONDecodeError:
            date_dict = ast.literal_eval(date_dict)

    # GeoJSON may store the dates as a dict {index: date_str}; extract values.
    if isinstance(date_dict, dict):
        try:
            keys = sorted(date_dict, key=lambda key: int(key))
        except (TypeError, ValueError):
            keys = list(date_dict)
        date_list = [date_dict[key] for key in keys]
    else:
        date_list = list(date_dict)

    days = []
    for d in date_list:
        d_str = str(d).strip()
        if not d_str or d_str in ("0", "nan", "None"):
            raise ValueError(
                f"Invalid date entry {d!r} in PASTIS-R metadata. "
                "Expected YYYYMMDD strings."
            )

        # Parse 8-digit YYYYMMDD (or 6-digit YYMMDD).
        if len(d_str) == 8:
            year = int(d_str[:4])
            month = int(d_str[4:6])
            day = int(d_str[6:])
        elif len(d_str) == 6:
            year = int(d_str[:2])
            month = int(d_str[2:4])
            day = int(d_str[4:])
            # PASTIS-R is 2019 data; assume 21st century for 2-digit years.
            if year < 50:
                year += 2000
            else:
                year += 1900
        else:
            raise ValueError(
                f"Unsupported PASTIS-R date format: {d!r}. Expected YYYYMMDD or YYMMDD."
            )

        try:
            d_date = datetime(year, month, day)
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse PASTIS-R date {d!r} as YYYY-MM-DD "
                f"({year:04d}-{month:02d}-{day:02d})."
            ) from exc

        days.append((d_date - reference_date).days)

    return torch.tensor(days, dtype=torch.long)


@DATASETS.register_module()
class UniverSatPASTISDataset(BaseDataset):
    """PASTIS-R segmentation dataset for UniverSat.

    Args:
        data_root: Root directory of the PASTIS-R dataset.
        modalities: List of modalities to load, e.g. ``["s2", "s1"]``.
        folds: List of folds to use (PASTIS-R has folds 1-5).
        reference_date: Reference date for computing relative day indices.
        norm_path: Directory holding ``NORM_{mod}_patch.json`` files. If None,
            no normalization is applied.
        temporal_dropout: Maximum number of timestamps kept per time series
            during training/val (set to a large value or inf to disable).
        pipeline: MMSegmentation transform pipeline.
        test_mode: Whether in test mode.
        meta_file: Name of the metadata file (default ``metadata.geojson``).
    """

    METAINFO = dict(classes=PASTIS_CLASSES, palette=PASTIS_PALETTE)

    def __init__(
        self,
        data_root: str,
        modalities: Sequence[str] = ("s2", "s1"),
        folds: Optional[Sequence[int]] = None,
        reference_date: str = "2018-01-01",
        norm_path: Optional[str] = None,
        temporal_dropout: int = 0,
        pipeline: Sequence = (),
        test_mode: bool = False,
        meta_file: str = "metadata.geojson",
        **kwargs,
    ):
        self.modalities = list(modalities)
        self.folds = list(folds) if folds is not None else None
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.temporal_dropout = (
            temporal_dropout if temporal_dropout > 0 else float("inf")
        )
        self.meta_file = meta_file
        self.test_mode = test_mode

        # Resolve a normalization directory relative to data_root. ``.`` is
        # the convenient way to select files stored directly under data_root.
        if norm_path in ("", "."):
            norm_path = data_root
        elif norm_path is not None and not os.path.isabs(norm_path):
            norm_path = os.path.join(data_root, norm_path)
        self.norm = load_norm(norm_path, self.modalities)

        super().__init__(
            data_root=data_root,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs,
        )

    def load_data_list(self) -> List[dict]:
        """Load lightweight metadata list from ``metadata.geojson``."""
        meta_path = os.path.join(self.data_root, self.meta_file)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        features = metadata.get("features")
        if not isinstance(features, list):
            raise ValueError(f"Expected a GeoJSON FeatureCollection in {meta_path}.")
        data_list = []
        for feature in features:
            row = feature.get("properties", {})
            fold = int(row["Fold"])
            if self.folds is not None and fold not in self.folds:
                continue
            info = {
                "id_patch": int(row["ID_PATCH"]),
                "fold": fold,
            }
            if "s2" in self.modalities:
                info["dates_s2"] = row["dates-S2"]
            if "s1" in self.modalities:
                info["dates_s1"] = row["dates-S1A"]
            data_list.append(info)
        return data_list

    def _load_modality(
        self, data_info: dict, modality: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load one modality array and its dates.

        Returns:
            (data, dates) where data is ``(T, C, H, W)`` and dates is ``(T,)``.
        """
        if modality == "s2":
            data_dir = "DATA_S2"
            prefix = "S2"
            date_key = "dates_s2"
        elif modality == "s1":
            data_dir = "DATA_S1A"
            prefix = "S1A"
            date_key = "dates_s1"
        else:
            raise ValueError(f"Unsupported modality for PASTIS-R: {modality}")

        id_patch = data_info["id_patch"]
        data_path = os.path.join(self.data_root, data_dir, f"{prefix}_{id_patch}.npy")
        data = torch.from_numpy(np.load(data_path).astype(np.float32))
        if data.ndim == 3:
            data = data.unsqueeze(0)
        if data.ndim != 4:
            raise ValueError(
                f"Expected {modality} data shape (T, C, H, W), got {tuple(data.shape)}"
            )

        dates = _prepare_dates(data_info[date_key], self.reference_date)
        if dates.shape[0] != data.shape[0]:
            raise ValueError(
                f"Modality {modality}: number of dates ({dates.shape[0]}) does not "
                f"match number of time steps ({data.shape[0]}) for patch {id_patch}."
            )
        return data, dates

    def _temporal_dropout(self, data: torch.Tensor, dates: torch.Tensor):
        """Randomly drop time steps at train/val time."""
        t = data.shape[0]
        if t > self.temporal_dropout:
            indices = torch.randperm(t)[: self.temporal_dropout].sort().values
            data = data[indices]
            dates = dates[indices]
        return data, dates

    def prepare_data(self, idx):
        """Load arrays, normalize, and run the pipeline."""
        if not self._fully_initialized:
            self.full_init()

        data_info = copy.deepcopy(self.get_data_info(idx))
        id_patch = data_info["id_patch"]

        # Load modalities and dates.
        output = {}
        for modality in self.modalities:
            data, dates = self._load_modality(data_info, modality)
            if not self.test_mode:
                data, dates = self._temporal_dropout(data, dates)
            output[modality] = data
            output[f"{modality}_dates"] = dates

        # Load dense label map.
        label_path = os.path.join(
            self.data_root, "ANNOTATIONS", f"TARGET_{id_patch}.npy"
        )
        label = np.load(label_path)
        if label.ndim == 3 and label.shape[0] == 1:
            label = label[0]
        if label.ndim != 2:
            raise ValueError(
                f"Expected TARGET_{id_patch}.npy to have shape (H,W) or "
                f"(1,H,W), got {tuple(label.shape)}."
            )
        output["gt_seg_map"] = torch.from_numpy(label.astype(np.int64))

        # Apply normalization in place.
        output = apply_norm(self.norm, output)

        # Attach lightweight metadata for the pipeline.
        output["id_patch"] = id_patch
        output["img_shape"] = tuple(output[self.modalities[0]].shape[-2:])
        output["ori_shape"] = output["img_shape"]
        output["pad_shape"] = output["img_shape"]

        return self.pipeline(output)


@FUNCTIONS.register_module()
def universat_pastis_collate(batch: List[dict]) -> dict:
    """Collate function that pads variable-length PASTIS time series.

    PASTIS-R samples may have different numbers of time steps (``T``). This
    collate pads each modality tensor and its dates to the maximum ``T`` in
    the batch before stacking. The final valid observation is repeated rather
    than inserting a fake zero-valued acquisition at the reference date.
    """
    if not batch:
        return {}

    # Collect all modality keys from the first sample.
    first_inputs = batch[0]["inputs"]
    modality_keys = list(first_inputs.keys())

    inputs = {}
    for key in modality_keys:
        tensors = [sample["inputs"][key] for sample in batch]
        # All tensors for the same key should have the same ndim except for
        # the time dimension (dim 0 for 4D tensors, dim 0 for 1D date tensors).
        if tensors[0].ndim == 4:
            max_t = max(t.shape[0] for t in tensors)
            padded = []
            for t in tensors:
                pad_size = max_t - t.shape[0]
                if pad_size > 0:
                    pad = t[-1:].expand(pad_size, *t.shape[1:])
                    t = torch.cat([t, pad], dim=0)
                padded.append(t)
            inputs[key] = torch.stack(padded, dim=0)
        elif tensors[0].ndim == 1:
            max_t = max(t.shape[0] for t in tensors)
            padded = []
            for t in tensors:
                pad_size = max_t - t.shape[0]
                if pad_size > 0:
                    pad = t[-1:].expand(pad_size)
                    t = torch.cat([t, pad], dim=0)
                padded.append(t)
            inputs[key] = torch.stack(padded, dim=0)
        else:
            inputs[key] = torch.stack(tensors, dim=0)

    data_samples = [sample["data_samples"] for sample in batch]
    return dict(inputs=inputs, data_samples=data_samples)
