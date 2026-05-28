from __future__ import annotations

from mmseg.datasets import PotsdamDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class OlmoEarthPotsdamDataset(PotsdamDataset):
    """Potsdam dataset that adds OLMoEarth project metadata.

    This keeps Potsdam on the normal OpenMMLab image-dataset path:
    ``LoadImageFromFile`` reads RGB tiles and ``LoadAnnotations`` reads label
    PNGs. The OLMoEarth-specific conversion happens later in the transform
    pipeline via ``RGBToOlmoEarthS2``.
    """

    RVSA_CLASSES = (
        "impervious_surface",
        "building",
        "low_vegetation",
        "tree",
        "car",
    )
    RVSA_PALETTE = [
        [255, 255, 255],
        [0, 0, 255],
        [0, 255, 255],
        [0, 255, 0],
        [255, 255, 0],
    ]
    OFFICIAL_TO_RVSA_LABEL_MAP = {
        0: 5,
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
    }

    def __init__(
        self,
        label_mapping: str = "official",
        **kwargs,
    ) -> None:
        self.label_mapping = label_mapping
        if label_mapping == "official":
            pass
        elif label_mapping == "official_to_rvsa_class5_ignore5":
            kwargs.setdefault("reduce_zero_label", False)
            kwargs.setdefault("ignore_index", 5)
            metainfo = dict(kwargs.pop("metainfo", {}) or {})
            metainfo.setdefault("classes", self.RVSA_CLASSES)
            metainfo.setdefault("palette", self.RVSA_PALETTE)
            kwargs["metainfo"] = metainfo
        else:
            raise ValueError(
                "label_mapping must be 'official' or "
                "'official_to_rvsa_class5_ignore5', got "
                f"{label_mapping!r}."
            )
        super().__init__(**kwargs)

    def load_data_list(self) -> list[dict]:
        data_list = super().load_data_list()
        for item in data_list:
            if self.label_mapping == "official_to_rvsa_class5_ignore5":
                item["label_map"] = dict(self.OFFICIAL_TO_RVSA_LABEL_MAP)
                item["reduce_zero_label"] = False
            item["dataset_name"] = "potsdam"
            item["olmoearth_modality"] = "rgb_to_sentinel2_l2a"
            item["olmoearth_num_timesteps"] = 1
        return data_list
