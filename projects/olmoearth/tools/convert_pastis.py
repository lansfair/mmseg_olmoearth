from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    label_stats,
    progress_iter,
    save_geotiff,
    save_json,
    save_timesteps_as_geotiffs,
    validate_labels,
)


NUM_CLASSES = 19
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
S1_BANDS = ["vv", "vh"]
EVAL_TO_OLMOEARTH_S2_BANDS = [1, 2, 3, 7, 4, 5, 6, 8, 11, 12, 0, 9]
EVAL_TO_OLMOEARTH_S1_BANDS = [0, 1]


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_timestamps(months: torch.Tensor) -> np.ndarray:
    values = []
    for month in months:
        item = int(month)
        month_value = int(str(item)[4:]) - 1
        year_value = int(str(item)[:4])
        values.append([1, month_value, year_value])
    return np.asarray(values, dtype=np.int64)


def _convert_split(
    input_root: Path,
    output_root: Path,
    split: str,
    ignore_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    split_dir = input_root / f"pastis_r_{split}"
    labels = torch.load(split_dir / "targets.pt", map_location="cpu")
    months = torch.load(split_dir / "months.pt", map_location="cpu")
    samples = []
    split_labels = []
    total = int(labels.shape[0])
    for idx in progress_iter(
        range(total),
        total=total,
        desc=f"{split}: converting PASTIS samples",
    ):
        sample_id = f"{split}_{idx:06d}"
        sample_dir = output_root / "samples" / sample_id
        _mkdir(sample_dir)

        s2 = torch.load(
            split_dir / "s2_images" / f"{idx}.pt",
            map_location="cpu",
        )
        s2 = s2[:, EVAL_TO_OLMOEARTH_S2_BANDS, :, :].numpy().astype(np.float32)
        s1 = torch.load(
            split_dir / "s1_images" / f"{idx}.pt",
            map_location="cpu",
        )
        s1 = s1[:, EVAL_TO_OLMOEARTH_S1_BANDS, :, :].numpy().astype(np.float32)
        label = labels[idx].numpy().astype(np.int64)
        label[label == -1] = ignore_index
        validate_labels(label, NUM_CLASSES, ignore_index, sample_id)

        s2_paths = save_timesteps_as_geotiffs(
            sample_dir,
            "sentinel2_l2a",
            s2,
            S2_BANDS,
        )
        save_timesteps_as_geotiffs(sample_dir, "sentinel1", s1, S1_BANDS)
        save_geotiff(sample_dir / "label.tif", label)
        timestamps = _to_timestamps(months[idx])
        split_labels.append(label)

        samples.append(
            {
                "sample_id": sample_id,
                "img_paths": [
                    f"samples/{sample_id}/{path}" for path in s2_paths
                ],
                "seg_map_path": f"samples/{sample_id}/label.tif",
                "timestamps": timestamps.tolist(),
                "olmoearth_modality": "sentinel2_l2a",
                "olmoearth_num_timesteps": int(s2.shape[0]),
            }
        )
    return samples, label_stats(split_labels, ignore_index, NUM_CLASSES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert processed OLMoEarth PASTIS-R tensors to MMSeg "
            "manifests."
        )
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Directory containing pastis_r_train/valid/test",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output data/olmoearth_mmseg/pastis directory",
    )
    parser.add_argument("--ignore-index", type=int, default=255)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    _mkdir(output_root)

    split_to_manifest = {"train": "train", "valid": "val", "test": "test"}
    splits = {}
    for source_split, target_split in split_to_manifest.items():
        samples, stats = _convert_split(
            input_root,
            output_root,
            source_split,
            args.ignore_index,
        )
        save_json(output_root / f"{target_split}.json", {"samples": samples})
        splits[target_split] = stats

    save_json(
        output_root / "metainfo.json",
        {
            "dataset": "pastis",
            "num_classes": NUM_CLASSES,
            "ignore_index": args.ignore_index,
            "image_layout": "img_paths_tif_tchw",
            "modalities": ["sentinel2_l2a", "sentinel1"],
            "bands": {
                "sentinel2_l2a": S2_BANDS,
                "sentinel1": S1_BANDS,
            },
            "normalization": {
                "source": "olmoearth_pretrain.data.normalize",
                "method": "computed",
            },
            "splits": splits,
        },
    )


if __name__ == "__main__":
    main()
