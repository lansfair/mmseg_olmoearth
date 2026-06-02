from pathlib import Path

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class DFC2020S2Dataset(BaseSegDataset):
    """DFC2020 S2 semantic segmentation dataset for MMSegmentation.

    The original Copernicus-Bench split CSVs store label filenames under
    ``dfc/``. Sentinel-2 images use the same filename with ``dfc`` replaced by
    ``s2``.
    """

    METAINFO = dict(
        classes=(
            "Forest",
            "Shrubland",
            "Grassland",
            "Wetland",
            "Cropland",
            "Urban/Built-up",
            "Barren",
            "Water",
        ),
        palette=[
            [0, 100, 0],
            [255, 187, 34],
            [255, 255, 76],
            [0, 150, 160],
            [240, 150, 255],
            [250, 0, 0],
            [180, 180, 180],
            [0, 100, 200],
        ],
    )

    def __init__(
        self,
        ann_file: str = "",
        data_prefix: dict = dict(img_path="s2", seg_map_path="dfc"),
        img_suffix=".tif",
        seg_map_suffix=".tif",
        ignore_index=255,
        reduce_zero_label=False,
        **kwargs,
    ):
        super().__init__(
            ann_file=ann_file,
            data_prefix=data_prefix,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=reduce_zero_label,
            **kwargs,
        )

    def load_data_list(self):
        img_dir = Path(self.data_prefix["img_path"])
        ann_dir = Path(self.data_prefix["seg_map_path"])
        data_list = []
        with open(self.ann_file, "r", encoding="utf-8") as f:
            for line in f:
                label_name = line.strip()
                if not label_name:
                    continue
                img_name = label_name.replace("dfc", "s2")
                data_list.append(
                    dict(
                        img_path=str(img_dir / img_name),
                        seg_map_path=str(ann_dir / label_name),
                        reduce_zero_label=self.reduce_zero_label,
                        seg_fields=[],
                    )
                )
        return data_list
