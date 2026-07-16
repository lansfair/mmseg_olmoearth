from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def ensure_split(
    metadata: pd.DataFrame,
    metadata_spec: dict[str, Any],
    split_spec: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    frame = metadata.copy()
    split_col = metadata_spec.get("split_column", "split")
    if split_col in frame and frame[split_col].notna().all():
        frame["_split"] = frame[split_col].astype(str).str.lower()
        unknown = set(frame["_split"]) - {"train", "val", "test"}
        if unknown:
            raise ValueError(f"Unknown split values: {sorted(unknown)}")
        return frame

    strategy = split_spec.get("strategy", "random")
    test_size = float(split_spec.get("test_size", 0.2))
    val_size = float(split_spec.get("val_size", 0.1))
    labels = frame[metadata_spec.get("label_column", "label")]

    if strategy == "spatial_block":
        lat_col = metadata_spec.get("latitude_column", "latitude")
        lon_col = metadata_spec.get("longitude_column", "longitude")
        block = float(split_spec.get("block_size_degrees", 1.0))
        if lat_col not in frame or lon_col not in frame:
            raise ValueError("spatial_block split requires latitude and longitude columns")
        lat_groups = np.floor(frame[lat_col].to_numpy() / block).astype(int).astype(str)
        lon_groups = np.floor(frame[lon_col].to_numpy() / block).astype(int).astype(str)
        groups = np.char.add(np.char.add(lat_groups, "_"), lon_groups)
        outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        remain_idx, test_idx = next(outer.split(frame, groups=groups))
        inner_fraction = val_size / (1.0 - test_size)
        inner = GroupShuffleSplit(n_splits=1, test_size=inner_fraction, random_state=seed + 1)
        train_rel, val_rel = next(inner.split(frame.iloc[remain_idx], groups=groups[remain_idx]))
        train_idx, val_idx = remain_idx[train_rel], remain_idx[val_rel]
    elif strategy == "random":
        all_idx = np.arange(len(frame))
        remain_idx, test_idx = train_test_split(
            all_idx, test_size=test_size, random_state=seed, stratify=labels
        )
        inner_fraction = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            remain_idx,
            test_size=inner_fraction,
            random_state=seed + 1,
            stratify=labels.iloc[remain_idx],
        )
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    frame["_split"] = "train"
    frame.loc[val_idx, "_split"] = "val"
    frame.loc[test_idx, "_split"] = "test"
    return frame
