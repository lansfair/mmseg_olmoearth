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

    def load_data_list(self) -> list[dict]:
        data_list = super().load_data_list()
        for item in data_list:
            item["dataset_name"] = "potsdam"
            item["olmoearth_modality"] = "rgb_to_sentinel2_l2a"
            item["olmoearth_num_timesteps"] = 1
        return data_list
