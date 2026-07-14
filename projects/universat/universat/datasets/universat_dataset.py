"""UniverSat-compatible segmentation dataset for MMSegmentation 1.x."""

import json
import os
from typing import List, Optional

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class UniverSatSegDataset(BaseSegDataset):
    """Custom segmentation dataset with multiple modality rasters.

    The annotation file is a JSON list of dicts::

        [
          {
            "filenames": {
              "s2": "s2/xxx.npy",
              "s1": "s1/xxx.npy"
            },
            "dates": {"s2": [365, 377, 389]},
            "date_filenames": {"s1": "dates/s1_xxx.npy"},
            "ann": {"seg_map": "masks/xxx.png"},
            "height": 360,
            "width": 360
          },
          ...
        ]

    Args:
        modalities: List of modality names. Must match the backbone.
        split: Path to a JSON split file. If provided, it overrides the
            directory-based discovery in ``BaseSegDataset``.
        *args, **kwargs: forwarded to ``BaseSegDataset``.
    """

    def __init__(
        self,
        modalities: List[str],
        split: Optional[str] = None,
        *args,
        **kwargs,
    ):
        self.modalities = list(modalities)
        self.split = split
        super().__init__(*args, **kwargs)

    def load_data_list(self):
        """Load data list from a JSON split file or directory structure."""
        if self.split is not None:
            split_path = self.split
            if self.data_root is not None:
                split_path = os.path.join(self.data_root, split_path)
            with open(split_path, "r", encoding="utf-8") as f:
                samples = json.load(f)

            data_list = []
            for sample in samples:
                modality_paths = sample.get("filenames", {})
                date_paths = sample.get("date_filenames", {})
                if self.data_root is not None:
                    modality_paths = {
                        mod: path
                        if os.path.isabs(path)
                        else os.path.join(self.data_root, path)
                        for mod, path in modality_paths.items()
                    }
                    date_paths = {
                        mod: path
                        if os.path.isabs(path)
                        else os.path.join(self.data_root, path)
                        for mod, path in date_paths.items()
                    }

                item = {
                    "modality_paths": modality_paths,
                    "modality_date_paths": date_paths,
                    "modality_dates": sample.get("dates", {}),
                    "height": sample.get("height", 0),
                    "width": sample.get("width", 0),
                    "label_map": self.label_map,
                    "reduce_zero_label": self.reduce_zero_label,
                    "seg_fields": [],
                }
                ann = sample.get("ann", {})
                seg_map = ann.get("seg_map")
                if seg_map is not None:
                    if self.data_root is not None and not os.path.isabs(seg_map):
                        seg_map = os.path.join(self.data_root, seg_map)
                    item["seg_map_path"] = seg_map
                data_list.append(item)
            return data_list

        # Fallback to directory-based discovery. This requires ``img_path`` to
        # point to a modality directory and may not work for all multimodal
        # layouts; the JSON split is the recommended format.
        return super().load_data_list()
