from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

TOOLS_DIR = Path(__file__).resolve().parents[2] / "olmoearth" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from common import (  # noqa: E402
    label_stats,
    progress_iter,
    save_geotiff,
    save_json,
    save_timesteps_as_geotiffs,
    validate_labels,
)


NUM_CLASSES = 19
SOURCE_IGNORE_VALUES = (-1, 19)
S2_10_BANDS = (
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
)


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sample_indices(num_available: int, num_timesteps: int) -> list[int]:
    if num_available <= 0:
        raise ValueError("PASTIS sample has no timesteps")
    n_dates = min(num_available, num_timesteps)
    indices = torch.linspace(0, num_available - 1, n_dates).long().tolist()
    while len(indices) < num_timesteps:
        indices.append(indices[-1])
    return indices


def _convert_fold_split(
    source_root: Path,
    output_root: Path,
    split: str,
    folds: tuple[int, ...],
    num_timesteps: int,
    ignore_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from torchgeo.datasets import PASTIS

    dataset = PASTIS(
        root=str(source_root),
        folds=folds,
        bands="s2",
        mode="semantic",
        download=False,
    )
    samples = []
    labels = []
    for idx in progress_iter(
        range(len(dataset)),
        total=len(dataset),
        desc=f"{split}: converting PASTIS folds {folds}",
    ):
        sample = dataset[idx]
        image = sample["image"].float()
        if int(image.shape[1]) != len(S2_10_BANDS):
            raise ValueError(
                f"Expected {len(S2_10_BANDS)} Sentinel-2 bands, "
                f"got {int(image.shape[1])} for sample {idx}"
            )
        label = sample["mask"].long().numpy().astype(np.int64).squeeze()
        for value in SOURCE_IGNORE_VALUES:
            label[label == value] = ignore_index
        sample_id = f"{split}_{idx:06d}"
        validate_labels(label, NUM_CLASSES, ignore_index, sample_id)

        indices = _sample_indices(int(image.shape[0]), num_timesteps)
        image = image[indices].numpy().astype(np.float32)

        sample_dir = output_root / "samples" / sample_id
        _mkdir(sample_dir)
        img_paths = save_timesteps_as_geotiffs(
            sample_dir,
            "sentinel2_10band",
            image,
            S2_10_BANDS,
        )
        save_geotiff(sample_dir / "label.tif", label)
        labels.append(label)
        samples.append(
            {
                "sample_id": sample_id,
                "img_paths": [
                    f"samples/{sample_id}/{path}" for path in img_paths
                ],
                "seg_map_path": f"samples/{sample_id}/label.tif",
                "pastis_num_timesteps": num_timesteps,
                "pastis_num_bands": len(S2_10_BANDS),
            }
        )
    return samples, label_stats(labels, ignore_index, NUM_CLASSES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw TorchGeo PASTIS to MMSeg manifests for "
            "DINOv3 temporal PASTIS training."
        )
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="TorchGeo PASTIS root, e.g. notebooks/01_pastis/data",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output directory for MMSeg manifests and GeoTIFF samples.",
    )
    parser.add_argument("--num-timesteps", type=int, default=12)
    parser.add_argument("--ignore-index", type=int, default=255)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    _mkdir(output_root)

    split_folds = {
        "train": (1, 2, 3),
        "val": (4,),
        "test": (5,),
    }
    stats_by_split = {}
    for split, folds in split_folds.items():
        samples, stats = _convert_fold_split(
            source_root,
            output_root,
            split,
            folds,
            args.num_timesteps,
            args.ignore_index,
        )
        save_json(output_root / f"{split}.json", {"samples": samples})
        stats_by_split[split] = stats

    save_json(
        output_root / "metainfo.json",
        {
            "dataset": "pastis_10band",
            "num_classes": NUM_CLASSES,
            "ignore_index": args.ignore_index,
            "image_layout": "img_paths_tif_tchw",
            "num_timesteps": args.num_timesteps,
            "bands": list(S2_10_BANDS),
            "normalization": "divide_by_10000_and_clip_0_1",
            "splits": stats_by_split,
        },
    )


if __name__ == "__main__":
    main()
