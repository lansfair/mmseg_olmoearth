from mmseg.datasets import BaseSegDataset


class CoBenchSegDataset(BaseSegDataset):
    """Common split-file helpers for Copernicus-Bench segmentation tasks."""

    def _data_info(self, img_path, seg_map_path):
        return dict(
            img_path=str(img_path),
            seg_map_path=str(seg_map_path),
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label,
            seg_fields=[],
        )
