from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload["samples"] if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise TypeError(
            f"Manifest must be a list or {{'samples': list}}: {path}"
        )
    return samples


def _load_metainfo(data_root: Path) -> dict[str, Any]:
    path = data_root / "metainfo.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"metainfo.json must contain a dict: {path}")
    return payload


def _resolve(data_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return data_root / path


def _load(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError(
                "Checking GeoTIFF manifests requires rasterio."
            ) from exc
        with rasterio.open(path) as src:
            array = src.read()
        if array.shape[0] == 1:
            return array[0]
        return array
    raise ValueError(f"Only GeoTIFF files are supported by manifests: {path}")


def _load_image(data_root: Path, sample: dict[str, Any]) -> np.ndarray:
    if "img_paths" not in sample:
        raise KeyError("Manifest sample must provide 'img_paths'.")
    images = [_load(_resolve(data_root, path)) for path in sample["img_paths"]]
    images = [image[None, ...] if image.ndim == 2 else image for image in images]
    return np.stack(images, axis=0)


def _validate_label_values(
    label: np.ndarray,
    metainfo: dict[str, Any],
    sample_id: str | None,
) -> None:
    if not metainfo:
        return
    num_classes = metainfo.get("num_classes")
    ignore_index = metainfo.get("ignore_index", 255)
    if num_classes is None:
        return
    invalid = (
        ((label < 0) | (label >= int(num_classes)))
        & (label != int(ignore_index))
    )
    if np.any(invalid):
        values = np.unique(label[invalid]).astype(int).tolist()
        raise ValueError(
            f"{sample_id} has invalid label values for "
            f"num_classes={num_classes}, ignore_index={ignore_index}: "
            f"{values}"
        )


def _check_sample(
    data_root: Path,
    sample: dict[str, Any],
    metainfo: dict[str, Any],
) -> dict[str, Any]:
    image = _load_image(data_root, sample)
    label = _load(_resolve(data_root, sample["seg_map_path"]))

    if image.ndim not in {3, 4}:
        raise ValueError(f"image must be 3D/4D, got {image.shape}")
    if label.squeeze().ndim != 2:
        raise ValueError(f"label must be 2D after squeeze, got {label.shape}")
    label_2d = label.squeeze()
    sample_id = sample.get("sample_id")
    _validate_label_values(label_2d, metainfo, sample_id)

    summary: dict[str, Any] = {
        "sample_id": sample_id,
        "image_shape": list(image.shape),
        "label_shape": list(label.shape),
        "label_min": int(label.min()) if label.size else None,
        "label_max": int(label.max()) if label.size else None,
        "label_unique": np.unique(label_2d).astype(int).tolist(),
    }

    valid_path = sample.get("valid_mask_path")
    if valid_path is not None:
        valid = _load(_resolve(data_root, valid_path)).squeeze()
        if valid.shape != label.squeeze().shape:
            raise ValueError(
                f"valid mask shape {valid.shape} != label shape "
                f"{label.squeeze().shape}"
            )
        summary["valid_pixels"] = int((valid > 0).sum())
        summary["total_pixels"] = int(valid.size)

    timestamps_value = sample.get("timestamps")
    if timestamps_value is not None:
        timestamps = np.asarray(timestamps_value)
        if timestamps.ndim != 2 or timestamps.shape[-1] != 3:
            raise ValueError(
                f"timestamps must be (T, 3), got {timestamps.shape}"
            )
        summary["timestamps_shape"] = list(timestamps.shape)
        summary["first_timestamp"] = timestamps[0].astype(int).tolist()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-check converted OLMoEarth MMSeg manifests and arrays."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    manifest_path = _resolve(data_root, args.ann_file)
    metainfo = _load_metainfo(data_root)
    samples = _load_manifest(manifest_path)
    if not samples:
        raise ValueError(f"Manifest contains no samples: {manifest_path}")

    checked = [
        _check_sample(data_root, sample, metainfo)
        for sample in samples[: args.max_samples]
    ]
    print(
        json.dumps(
            {
                "num_samples": len(samples),
                "metainfo": metainfo,
                "checked": checked,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
