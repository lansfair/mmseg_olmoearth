from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.transform import from_origin

from projects.geofm_embeddings.evaluation.clustering import (
    evaluate_kmeans,
)
from projects.geofm_embeddings.evaluation.io import load_embedding
from projects.geofm_embeddings.evaluation.preprocessing import l2
from projects.geofm_embeddings.evaluation.retrieval import (
    evaluate_semantic_retrieval,
)
from projects.geofm_embeddings.evaluation.runner import (
    run_experiment,
)
from projects.geofm_embeddings.evaluation.supervised import (
    evaluate_knn,
    evaluate_linear_probe,
)


def _synthetic_data():
    rng = np.random.default_rng(7)
    labels = np.repeat(np.asarray(["a", "b", "c"]), 20)
    centers = np.eye(3, 8, dtype=np.float32) * 7
    values = centers[np.repeat(np.arange(3), 20)]
    values += rng.normal(scale=0.2, size=values.shape)
    split = np.tile(np.asarray(["train"] * 14 + ["test"] * 6), 3)
    ids = np.asarray([f"sample_{index:03d}" for index in range(len(labels))])
    return values.astype(np.float32), labels, split, ids


def test_sample_level_evaluation_modules() -> None:
    values, labels, split, ids = _synthetic_data()
    normalized = l2(values)
    clustering = evaluate_kmeans(
        normalized, labels, {"cluster_counts": ["classes"], "n_init": 5}, [1, 2]
    )
    knn = evaluate_knn(
        normalized, labels, split, {"k_values": [3], "budgets_per_class": [5]}, [1]
    )
    linear = evaluate_linear_probe(
        values,
        labels,
        split,
        {"budgets_per_class": [5], "c_values": [0.1, 1], "cv_folds": 2},
        [1],
    )
    retrieval, details = evaluate_semantic_retrieval(
        normalized,
        labels,
        ids,
        split,
        {"gallery_split": "train", "query_split": "test", "k_values": [1, 5]},
    )
    assert np.mean([row["ari"] for row in clustering]) > 0.9
    assert knn[0]["macro_f1"] > 0.9
    assert linear[0]["macro_f1"] > 0.9
    assert retrieval[0]["hit_rate_at_1"] > 0.9
    assert len(details) == int(np.sum(split == "test"))


def test_geofm_manifest_loads_variable_size_dense_geotiffs(tmp_path) -> None:
    records = []
    for index, shape in enumerate(((3, 4), (5, 2))):
        sample_id = f"dense_{index}"
        relative = f"test/{sample_id}/embedding.tif"
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        array = np.stack([
            np.full(shape, index + 1, dtype=np.float32),
            np.full(shape, index + 2, dtype=np.float32),
        ])
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=shape[0],
            width=shape[1],
            count=2,
            dtype="float32",
            transform=from_origin(0, shape[0], 1, 1),
        ) as destination:
            destination.write(array)
        records.append({"sample_id": sample_id, "embedding_path": relative})
    (tmp_path / "test.json").write_text(
        json.dumps({
            "format": "geofm_embedding_manifest_v1",
            "split": "test",
            "count": len(records),
            "samples": records,
        }),
        encoding="utf-8",
    )
    table = load_embedding({
        "name": "dense_model",
        "format": "geofm_manifests",
        "path": str(tmp_path),
        "splits": ["test"],
        "pooling": "stats",
    }, tmp_path)
    assert table.values.shape == (2, 8)
    assert table.sample_ids.tolist() == ["dense_0", "dense_1"]


def test_manifest_to_runner_end_to_end(tmp_path) -> None:
    values, labels, split, ids = _synthetic_data()
    export_root = tmp_path / "export"
    records_by_split = {"train": [], "test": []}
    for value, label, split_name, sample_id in zip(values, labels, split, ids):
        relative = f"{split_name}/{sample_id}/embedding.pt"
        path = export_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(value), path)
        records_by_split[split_name].append({
            "sample_id": sample_id,
            "embedding_path": relative,
        })
    for split_name, records in records_by_split.items():
        (export_root / f"{split_name}.json").write_text(
            json.dumps({
                "format": "geofm_embedding_manifest_v1",
                "split": split_name,
                "count": len(records),
                "samples": records,
            }),
            encoding="utf-8",
        )
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"sample_id": ids, "label": labels, "split": split}).to_csv(
        metadata_path, index=False
    )
    config = {
        "experiment_id": "manifest_smoke",
        "output_dir": str(tmp_path / "results"),
        "random_seed": 3,
        "seeds": [3],
        "metadata": {
            "path": str(metadata_path),
            "id_column": "sample_id",
            "label_column": "label",
            "split_column": "split",
        },
        "models": [{
            "name": "exported_model",
            "format": "geofm_manifests",
            "path": str(export_root),
            "splits": ["train", "test"],
        }],
        "tracks": [{"name": "native", "pca_dim": None}],
        "tasks": {
            "clustering": {
                "enabled": True,
                "max_samples": 60,
                "dbscan_max_samples": 60,
                "kmeans": {"cluster_counts": ["classes"], "n_init": 2},
                "dbscan": {"min_samples": [3], "eps_multipliers": [1.0]},
            },
            "knn": {
                "enabled": True,
                "k_values": [3],
                "budgets_per_class": [5],
            },
            "linear_probe": {
                "enabled": True,
                "budgets_per_class": [5],
                "c_values": [0.1, 1.0],
                "cv_folds": 2,
            },
            "retrieval": {
                "enabled": True,
                "gallery_split": "train",
                "query_split": "test",
                "k_values": [1, 5],
            },
            "intrinsic_dimension": {"enabled": False},
            "visualization": {"enabled": True, "max_samples": 60},
        },
    }
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    run_dir = run_experiment(config_path)
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "complete"
    assert (run_dir / "retrieval.csv").exists()
    assert (run_dir / "retrieval_details_exported_model_native.csv").exists()
    assert (run_dir / "figures" / "pca_exported_model_native.png").exists()
