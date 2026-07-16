from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _sample_for_plot(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(labels) <= maximum:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    per_class = max(1, maximum // len(np.unique(labels)))
    selected = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        count = min(per_class, len(candidates))
        selected.extend(rng.choice(candidates, count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def plot_pca_diagnostics(
    values: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    sample_ids: np.ndarray,
    output_path: Path,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    if values.shape[1] < 2:
        raise ValueError("PCA visualization requires at least two embedding dimensions")
    train = split == "train"
    if train.sum() < 3:
        raise ValueError("PCA visualization requires at least three training samples")
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(values[train])
    coordinates = pca.transform(values)
    selected = _sample_for_plot(
        labels, int(config.get("max_samples", 5000)), seed
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    classes = np.unique(labels[selected])
    color_map = plt.get_cmap("tab20" if len(classes) <= 20 else "hsv")
    for index, label in enumerate(classes):
        mask = selected[labels[selected] == label]
        axes[0].scatter(
            coordinates[mask, 0], coordinates[mask, 1], s=10, alpha=0.6,
            color=color_map(index / max(len(classes) - 1, 1)), label=str(label),
        )
    axes[0].set_title("PCA by class")
    if len(classes) <= 20:
        axes[0].legend(fontsize=7, ncol=2, markerscale=1.5)

    markers = {"train": "o", "val": "^", "test": "s"}
    for split_name in np.unique(split[selected]):
        mask = selected[split[selected] == split_name]
        axes[1].scatter(
            coordinates[mask, 0], coordinates[mask, 1], s=10, alpha=0.55,
            marker=markers.get(str(split_name), "o"), label=str(split_name),
        )
    axes[1].set_title("PCA by split")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        axis.grid(alpha=0.2)
    figure.savefig(output_path, dpi=int(config.get("dpi", 180)))
    plt.close(figure)

    pd.DataFrame({
        "sample_id": sample_ids[selected],
        "label": labels[selected],
        "split": split[selected],
        "pc1": coordinates[selected, 0],
        "pc2": coordinates[selected, 1],
    }).to_csv(output_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    report = {
        "samples_plotted": len(selected),
        "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_variance": float(pca.explained_variance_ratio_[1]),
        "total_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
