from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from common import label_stats, progress_iter, save_json, validate_labels


POTSDAM_CLASSES = (
    "impervious_surface",
    "building",
    "low_vegetation",
    "tree",
    "car",
    "clutter",
)
POTSDAM_PALETTE = [
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
    [255, 0, 0],
]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
LABEL_SUFFIXES = IMAGE_SUFFIXES | {".npy"}


def _read_image(path: Path) -> np.ndarray:
    from PIL import Image

    image = np.asarray(Image.open(path).convert("RGB"))
    return image.astype(np.float32, copy=False)


def _read_label(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        label = np.load(path)
    else:
        from PIL import Image

        label = np.asarray(Image.open(path))
    if label.ndim == 3:
        label = _color_label_to_index(label)
    return label.squeeze().astype(np.int64, copy=False)


def _color_label_to_index(label: np.ndarray) -> np.ndarray:
    color_map = np.asarray(
        [[0, 0, 0], *POTSDAM_PALETTE],
        dtype=np.uint8,
    )
    output = np.zeros(label.shape[:2], dtype=np.int64)
    rgb = label[..., :3].astype(np.uint8, copy=False)
    for class_idx, color in enumerate(color_map):
        output[np.all(rgb == color, axis=-1)] = class_idx
    return output


def _reduce_zero_label(label: np.ndarray, ignore_index: int) -> np.ndarray:
    out = label.copy()
    out[out == 0] = ignore_index
    out = out - 1
    out[out == ignore_index - 1] = ignore_index
    return out


def _find_splits(input_root: Path) -> list[str]:
    splits = []
    for split in ("train", "val", "test"):
        if (input_root / "img_dir" / split).is_dir():
            splits.append(split)
    if not splits:
        raise FileNotFoundError(
            "Expected mmseg-style Potsdam directories such as "
            f"{input_root / 'img_dir' / 'train'}"
        )
    return splits


def _list_images(img_dir: Path) -> list[Path]:
    return sorted(
        path for path in img_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _match_label(ann_dir: Path, image_path: Path) -> Path:
    for suffix in LABEL_SUFFIXES:
        candidate = ann_dir / f"{image_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No matching label for {image_path.name} under {ann_dir}"
    )


def _convert_split(
    input_root: Path,
    output_root: Path,
    split: str,
    ignore_index: int,
    reduce_zero_label: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    img_dir = input_root / "img_dir" / split
    ann_dir = input_root / "ann_dir" / split
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Missing annotation directory: {ann_dir}")

    samples: list[dict[str, Any]] = []
    split_labels: list[np.ndarray] = []
    image_paths = _list_images(img_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {img_dir}")

    for idx, image_path in enumerate(
        progress_iter(
            image_paths,
            total=len(image_paths),
            desc=f"{split}: converting Potsdam samples",
        )
    ):
        label_path = _match_label(ann_dir, image_path)
        image = _read_image(image_path)
        label = _read_label(label_path)
        if reduce_zero_label:
            label = _reduce_zero_label(label, ignore_index)
        validate_labels(
            label,
            len(POTSDAM_CLASSES),
            ignore_index,
            image_path.stem,
        )

        sample_id = f"{split}_{idx:06d}"
        sample_dir = output_root / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.save(sample_dir / "rgb.npy", image)
        np.save(sample_dir / "label.npy", label)
        split_labels.append(label)

        samples.append(
            {
                "sample_id": sample_id,
                "img_path": f"samples/{sample_id}/rgb.npy",
                "seg_map_path": f"samples/{sample_id}/label.npy",
                "olmoearth_modality": "rgb_to_sentinel2_l2a",
                "olmoearth_num_timesteps": 1,
                "dataset_name": "potsdam",
                "source": {
                    "image_path": str(image_path.relative_to(input_root)),
                    "label_path": str(label_path.relative_to(input_root)),
                    "raw_image_shape": list(image.shape),
                    "raw_label_shape": list(label.shape),
                },
            }
        )

    stats = label_stats(split_labels, ignore_index, len(POTSDAM_CLASSES))
    return samples, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert mmseg-prepared Potsdam tiles into the OLMoEarth MMSeg "
            "manifest format. Run tools/dataset_converters/potsdam.py first "
            "if you only have the original ISPRS zip files."
        )
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Directory with img_dir/{train,val} and ann_dir/{train,val}.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.set_defaults(reduce_zero_label=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--reduce-zero-label",
        dest="reduce_zero_label",
        action="store_true",
        help=(
            "Map official Potsdam 0/black boundary to ignore and "
            "1..6 to 0..5."
        ),
    )
    group.add_argument(
        "--no-reduce-zero-label",
        dest="reduce_zero_label",
        action="store_false",
        help="Use this if labels are already encoded as class ids 0..5.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_stats: dict[str, dict[str, Any]] = {}
    for split in _find_splits(input_root):
        samples, stats = _convert_split(
            input_root=input_root,
            output_root=output_root,
            split=split,
            ignore_index=args.ignore_index,
            reduce_zero_label=args.reduce_zero_label,
        )
        save_json(output_root / f"{split}.json", {"samples": samples})
        split_stats[split] = stats

    save_json(
        output_root / "metainfo.json",
        {
            "dataset": "potsdam",
            "num_classes": len(POTSDAM_CLASSES),
            "classes": POTSDAM_CLASSES,
            "palette": POTSDAM_PALETTE,
            "ignore_index": args.ignore_index,
            "image_layout": "HWC",
            "modalities": ["rgb_to_sentinel2_l2a"],
            "bands": ["R", "G", "B"],
            "target_olmoearth_modality": "sentinel2_l2a",
            "target_olmoearth_bands": ["B04", "B03", "B02"],
            "rgb_input_value_range": "0_255",
            "reduce_zero_label": args.reduce_zero_label,
            "splits": split_stats,
        },
    )


if __name__ == "__main__":
    main()
