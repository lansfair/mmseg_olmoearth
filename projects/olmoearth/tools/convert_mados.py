from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import (
    label_stats,
    progress_iter,
    save_geotiff,
    save_json,
    save_timesteps_as_geotiffs,
    validate_labels,
)


NUM_CLASSES = 15
IGNORE_INDEX = 255
S2_BANDS = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
]
EVAL_TO_OLMOEARTH_S2_BANDS = [1, 2, 3, 7, 4, 5, 6, 8, 11, 12, 0, 9]


def _convert_split(
    input_root: Path,
    output_root: Path,
    split: str,
    manifest_name: str,
) -> dict:
    import torch

    obj = torch.load(input_root / f"MADOS_{split}.pt", map_location="cpu")
    images = obj["images"]
    labels = obj["labels"]
    samples = []
    split_labels = []
    total = int(images.shape[0])
    for idx in progress_iter(
        range(total),
        total=total,
        desc=f"{manifest_name}: converting MADOS samples",
    ):
        sample_id = f"{manifest_name}_{idx:06d}"
        sample_dir = output_root / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        image = images[idx].numpy().astype(np.float32)
        if image.ndim != 3:
            raise ValueError(
                f"Expected MADOS image with 3 dimensions, got {image.shape}"
            )
        if image.shape[0] == 13:
            image = np.moveaxis(image, 0, -1)
        if image.shape[-1] != 13:
            raise ValueError(
                f"Expected 13 Sentinel-2 channels, got {image.shape}"
            )
        image = image[:, :, EVAL_TO_OLMOEARTH_S2_BANDS]
        image = image.transpose(2, 0, 1)[None, ...]
        label = labels[idx].numpy().astype(np.int64)
        validate_labels(label, NUM_CLASSES, IGNORE_INDEX, sample_id)
        timestamps = np.asarray([[1, 6, 2020]], dtype=np.int64)
        image_paths = save_timesteps_as_geotiffs(
            sample_dir,
            "sentinel2_l2a",
            image,
            S2_BANDS,
        )
        save_geotiff(sample_dir / "label.tif", label)
        split_labels.append(label)
        samples.append(
            {
                "sample_id": sample_id,
                "img_paths": [
                    f"samples/{sample_id}/{path}" for path in image_paths
                ],
                "seg_map_path": f"samples/{sample_id}/label.tif",
                "timestamps": timestamps.tolist(),
                "olmoearth_modality": "sentinel2_l2a",
                "olmoearth_num_timesteps": 1,
            }
        )
    save_json(output_root / f"{manifest_name}.json", {"samples": samples})
    return label_stats(split_labels, IGNORE_INDEX, NUM_CLASSES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert processed MADOS tensors to MMSeg manifests."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in ["train", "valid", "test"]:
        manifest = "val" if split == "valid" else split
        splits[manifest] = _convert_split(
            input_root,
            output_root,
            split,
            manifest,
        )
    save_json(
        output_root / "metainfo.json",
        {
            "dataset": "mados",
            "num_classes": NUM_CLASSES,
            "ignore_index": IGNORE_INDEX,
            "image_layout": "img_paths_tif_tchw",
            "modalities": ["sentinel2_l2a"],
            "bands": S2_BANDS,
            "normalization": {
                "source": "olmoearth_pretrain.evals.datasets.mados_dataset",
                "method": "norm_no_clip_2_std",
            },
            "timestamps": {"default_day_month_year": [1, 6, 2020]},
            "splits": splits,
        },
    )


if __name__ == "__main__":
    main()
