from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import label_stats, save_json, validate_labels


NUM_CLASSES = 2
IGNORE_INDEX = 255
S1_BANDS = ["vv", "vh"]


def _remove_nan(
    s1: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    keep_s1, keep_labels = [], []
    for idx in range(s1.shape[0]):
        if torch.any(torch.isnan(s1[idx])) or torch.any(torch.isinf(s1[idx])):
            continue
        keep_s1.append(s1[idx])
        keep_labels.append(labels[idx])
    return torch.stack(keep_s1), torch.stack(keep_labels)


def _convert_split(
    input_root: Path,
    output_root: Path,
    split: str,
    manifest_name: str,
) -> dict:
    import torch

    obj = torch.load(input_root / f"flood_{split}_data.pt", map_location="cpu")
    raw_count = int(obj["s1"].shape[0])
    s1, labels = _remove_nan(obj["s1"], obj["labels"])
    samples = []
    split_labels = []
    for idx in range(s1.shape[0]):
        sample_id = f"{manifest_name}_{idx:06d}"
        sample_dir = output_root / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        image = s1[idx].unsqueeze(0).numpy().astype(np.float32)
        label = labels[idx][0].numpy().astype(np.int64)
        validate_labels(label, NUM_CLASSES, IGNORE_INDEX, sample_id)
        timestamps = np.asarray([[1, 6, 2020]], dtype=np.int64)
        np.save(sample_dir / "sentinel1.npy", image)
        np.save(sample_dir / "label.npy", label)
        np.save(sample_dir / "timestamps.npy", timestamps)
        split_labels.append(label)
        samples.append(
            {
                "sample_id": sample_id,
                "img_path": f"samples/{sample_id}/sentinel1.npy",
                "seg_map_path": f"samples/{sample_id}/label.npy",
                "timestamps_path": f"samples/{sample_id}/timestamps.npy",
                "olmoearth_modality": "sentinel1",
                "olmoearth_num_timesteps": 1,
            }
        )
    save_json(output_root / f"{manifest_name}.json", {"samples": samples})
    stats = label_stats(split_labels, IGNORE_INDEX, NUM_CLASSES)
    stats["raw_samples_before_nan_filter"] = raw_count
    stats["removed_nan_or_inf_samples"] = raw_count - len(split_labels)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert processed Sen1Floods11 tensors to MMSeg manifests."
        )
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    split_names = [("train", "train"), ("valid", "val"), ("test", "test")]
    splits = {}
    for source_split, manifest_name in split_names:
        splits[manifest_name] = _convert_split(
            input_root,
            output_root,
            source_split,
            manifest_name,
        )
    save_json(
        output_root / "metainfo.json",
        {
            "dataset": "sen1floods11",
            "num_classes": NUM_CLASSES,
            "ignore_index": IGNORE_INDEX,
            "image_layout": "TCHW",
            "modalities": ["sentinel1"],
            "bands": S1_BANDS,
            "normalization": {
                "source": "olmoearth_pretrain.evals.datasets.floods_dataset",
                "method": "norm_no_clip_2_std",
            },
            "timestamps": {"default_day_month_year": [1, 6, 2020]},
            "splits": splits,
        },
    )


if __name__ == "__main__":
    main()
