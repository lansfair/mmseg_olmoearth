from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
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


@dataclass(frozen=True)
class RslearnSegSpec:
    dataset_name: str
    num_classes: int
    nodata_value: int
    patch_size: int
    time_year: int
    split_tags: dict[str, str]
    classes: tuple[str, ...]


DATASET_SPECS = {
    "awf": RslearnSegSpec(
        dataset_name="awf",
        num_classes=10,
        nodata_value=9,
        patch_size=4,
        time_year=2024,
        split_tags={"train": "train", "val": "val", "test": "val"},
        classes=(
            "woodland_forest",
            "open_water",
            "shrubland_savanna",
            "herbaceous_wetland",
            "grassland_barren",
            "agriculture_settlement",
            "montane_forest",
            "lava_forest",
            "urban_dense_development",
            "nodata",
        ),
    ),
    "nandi": RslearnSegSpec(
        dataset_name="nandi",
        num_classes=11,
        nodata_value=10,
        patch_size=1,
        time_year=2024,
        split_tags={"train": "train", "val": "val", "test": "val"},
        classes=(
            "coffee",
            "grassland",
            "trees",
            "maize",
            "sugarcane",
            "tea",
            "vegetables",
            "legumes",
            "water",
            "builtup",
            "nodata",
        ),
    ),
}


def _import_rslearn():
    from rslearn.config import DType
    from rslearn.dataset import Dataset
    from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
    from rslearn.train.tasks.multi_task import MultiTask
    from rslearn.train.tasks.segmentation import SegmentationTask
    from upath import UPath

    return (
        DType,
        Dataset,
        DataInput,
        ModelDataset,
        MultiTask,
        SegmentationTask,
        SplitConfig,
        UPath,
    )


def _build_split_config(split_tag: str):
    imports = _import_rslearn()
    SplitConfig = imports[6]
    return SplitConfig(groups=["spatial_split"], tags={"split": split_tag})


def _build_rslearn_dataset(
    dataset_path: Path,
    spec: RslearnSegSpec,
    split_tag: str,
    workers: int,
):
    (
        DType,
        Dataset,
        DataInput,
        ModelDataset,
        MultiTask,
        SegmentationTask,
        _,
        UPath,
    ) = _import_rslearn()
    kwargs = {
        "dataset": Dataset(path=UPath(dataset_path)),
        "split_config": _build_split_config(split_tag),
        "inputs": {
            "sentinel2_l2a": DataInput(
                data_type="raster",
                layers=["sentinel2"],
                bands=S2_BANDS,
                passthrough=True,
                dtype=DType.FLOAT32,
                load_all_item_groups=True,
                load_all_layers=True,
            ),
            "label": DataInput(
                data_type="raster",
                layers=["label"],
                bands=["category"],
                is_target=True,
                dtype=DType.INT32,
            ),
        },
        "task": MultiTask(
            tasks={
                "segment": SegmentationTask(
                    num_classes=spec.num_classes,
                    zero_is_invalid=False,
                    nodata_value=spec.nodata_value,
                    metric_kwargs={"average": "micro"},
                )
            },
            input_mapping={"segment": {"label": "targets"}},
        ),
        "workers": workers,
        "fix_crop_pick": True,
    }
    return ModelDataset(**kwargs)


def _to_numpy(value: Any) -> np.ndarray:
    import torch

    if hasattr(value, "image"):
        value = value.image
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().cpu().numpy()


def _time_ranges(value: Any):
    return getattr(value, "timestamps", None)


def _legacy_timestamps(num_timesteps: int, year: int) -> np.ndarray:
    return np.asarray(
        [[1, month, year] for month in range(num_timesteps)],
        dtype=np.int64,
    )


def _actual_timestamps(time_ranges) -> np.ndarray | None:
    if not time_ranges:
        return None
    values = []
    for start, end in time_ranges:
        midpoint = start + (end - start) / 2
        if not isinstance(midpoint, datetime):
            return None
        values.append([midpoint.day, midpoint.month - 1, midpoint.year])
    return np.asarray(values, dtype=np.int64)


def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    out: dict[str, Any] = {}
    for name in (
        "window_group",
        "window_name",
        "window_bounds",
        "crop_bounds",
        "patch_idx",
        "num_patches_in_window",
        "num_crops_in_window",
        "time_range",
        "dataset_source",
    ):
        if not hasattr(metadata, name):
            continue
        value = getattr(metadata, name)
        if value is None:
            out[name] = None
        elif name == "time_range":
            out[name] = [value[0].isoformat(), value[1].isoformat()]
        elif isinstance(value, tuple):
            out[name] = list(value)
        else:
            out[name] = value
    return out


def _convert_split(
    rslearn_dataset,
    output_root: Path,
    manifest_name: str,
    spec: RslearnSegSpec,
    timestamp_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    split_labels = []

    total = len(rslearn_dataset)
    for idx in progress_iter(
        range(total),
        total=total,
        desc=f"{manifest_name}: converting rslearn samples",
    ):
        input_dict, target_dict, metadata = rslearn_dataset[idx]
        raster = input_dict["sentinel2_l2a"]
        image = _to_numpy(raster).astype(np.float32)
        if image.ndim != 4:
            raise ValueError(
                f"Expected sentinel2_l2a CTHW image, got {image.shape}"
            )
        label = _to_numpy(
            target_dict["segment"]["classes"]
        ).squeeze().astype(np.int64)
        valid = _to_numpy(
            target_dict["segment"]["valid"]
        ).squeeze().astype(np.float32)
        validate_labels(label, spec.num_classes, 255, f"{manifest_name}_{idx}")

        if timestamp_mode == "actual":
            timestamps = _actual_timestamps(_time_ranges(raster))
            if timestamps is None:
                timestamps = _legacy_timestamps(image.shape[1], spec.time_year)
        elif timestamp_mode == "legacy":
            timestamps = _legacy_timestamps(image.shape[1], spec.time_year)
        else:
            raise ValueError("timestamp_mode must be 'legacy' or 'actual'")

        sample_id = f"{manifest_name}_{idx:06d}"
        sample_dir = output_root / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        image_paths = save_timesteps_as_geotiffs(
            sample_dir,
            "sentinel2_l2a",
            image.transpose(1, 0, 2, 3),
            S2_BANDS,
        )
        save_geotiff(sample_dir / "label.tif", label)
        save_geotiff(sample_dir / "valid_mask.tif", valid.astype(np.uint8))
        split_labels.append(label)

        samples.append(
            {
                "sample_id": sample_id,
                "img_paths": [
                    f"samples/{sample_id}/{path}" for path in image_paths
                ],
                "seg_map_path": f"samples/{sample_id}/label.tif",
                "valid_mask_path": f"samples/{sample_id}/valid_mask.tif",
                "timestamps": timestamps.tolist(),
                "olmoearth_modality": "sentinel2_l2a",
                "olmoearth_num_timesteps": int(image.shape[1]),
                "dataset_name": spec.dataset_name,
                "rslearn": {
                    **_metadata_to_dict(metadata),
                    "source_index": idx,
                    "raw_shape": list(image.shape),
                    "timestamp_mode": timestamp_mode,
                },
            }
        )
    return samples, label_stats(split_labels, 255, spec.num_classes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert rslearn segmentation datasets into the OLMoEarth MMSeg "
            "manifest format. Dataset-specific mappings should be implemented "
            "explicitly so label, valid-mask, timestamp, and band semantics "
            "are reviewable before training."
        )
    )
    parser.add_argument("--dataset", required=True, choices=["awf", "nandi"])
    parser.add_argument(
        "--input-root",
        required=True,
        help="rslearn dataset path",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--timestamp-mode",
        choices=["legacy", "actual"],
        default="legacy",
        help=(
            "legacy matches rslearn OlmoEarth default dummy month "
            "timestamps."
        ),
    )
    args = parser.parse_args()

    spec = DATASET_SPECS[args.dataset]
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_stats: dict[str, dict[str, Any]] = {}
    for manifest_name, split_tag in spec.split_tags.items():
        rslearn_dataset = _build_rslearn_dataset(
            dataset_path=input_root,
            spec=spec,
            split_tag=split_tag,
            workers=args.workers,
        )
        samples, stats = _convert_split(
            rslearn_dataset=rslearn_dataset,
            output_root=output_root,
            manifest_name=manifest_name,
            spec=spec,
            timestamp_mode=args.timestamp_mode,
        )
        save_json(output_root / f"{manifest_name}.json", {"samples": samples})
        split_stats[manifest_name] = stats

    save_json(
        output_root / "metainfo.json",
        {
            "dataset": spec.dataset_name,
            "num_classes": spec.num_classes,
            "classes": spec.classes,
            "nodata_value": spec.nodata_value,
            "ignore_index": 255,
            "image_layout": "img_paths_tif_tchw",
            "modalities": ["sentinel2_l2a"],
            "bands": S2_BANDS,
            "rslearn_pad_size": 31,
            "rslearn_crop_size": 16,
            "patch_size": spec.patch_size,
            "splits": split_stats,
        },
    )


if __name__ == "__main__":
    main()
