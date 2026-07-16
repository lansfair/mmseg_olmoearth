from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.geofm_embeddings.evaluation.io import geotiff_sample_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build scene-level metadata from embedding and label GeoTIFFs."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--embedding-filename", default="embedding.tif")
    parser.add_argument("--label-filename", default="label.tif")
    parser.add_argument(
        "--id-mode",
        choices=("parent", "relative_parent", "relative_file"),
        default="parent",
    )
    parser.add_argument("--ignore-label", type=int, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rows = []
    for embedding_path in sorted(root.rglob(args.embedding_filename)):
        label_path = embedding_path.with_name(args.label_filename)
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        with rasterio.open(label_path) as source:
            label_array = source.read().reshape(-1)
        valid = label_array[~np.isin(label_array, args.ignore_label)]
        if not len(valid):
            raise ValueError(f"No valid labels in {label_path}")
        labels, counts = np.unique(valid, return_counts=True)
        relative = embedding_path.relative_to(root)
        split = relative.parts[0].lower() if len(relative.parts) > 1 else "unspecified"
        if split == "valid":
            split = "val"
        rows.append({
            "sample_id": geotiff_sample_id(embedding_path, root, args.id_mode),
            "label": int(labels[np.argmax(counts)]),
            "split": split,
            "label_classes_in_raster": len(labels),
            "majority_label_fraction": float(counts.max() / counts.sum()),
            "embedding_path": relative.as_posix(),
        })
    if not rows:
        raise FileNotFoundError(f"No {args.embedding_filename} found under {root}")
    frame = pd.DataFrame(rows)
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample IDs: {duplicates[:10]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(frame.groupby(["split", "label"]).size(), flush=True)
    print(f"Saved {len(frame)} rows to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
