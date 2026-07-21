from __future__ import annotations

import json

import torch

from projects.geofm_embeddings.evaluation.bundle import (
    bundle_paths,
    load_bundle_split,
)
from projects.geofm_embeddings.evaluation.bundle_tasks import (
    gather_features,
    run_cosine_retrieval,
    run_dbscan,
    run_kmeans,
)
from projects.geofm_embeddings.evaluation.linear import run_linear
from projects.geofm_embeddings.evaluation.knn import run_knn


def _directory(root, dataset="example", model="model"):
    path = root / dataset / model
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_dense_group(root) -> None:
    directory = _directory(root)
    labels = torch.arange(8).view(1, 1, 8).expand(4, 8, 8) % 4
    embeddings = torch.nn.functional.one_hot(labels, num_classes=4).float()
    for split in ("train", "valid", "test"):
        torch.save(
            {"embeddings": embeddings, "labels": labels}, directory / f"{split}.pt"
        )


def _write_global_group(root) -> None:
    directory = _directory(root, dataset="scenes")
    labels = torch.arange(40) % 4
    embeddings = torch.nn.functional.one_hot(labels, num_classes=4).float()
    for split in ("train", "valid", "test"):
        torch.save(
            {"embeddings": embeddings, "labels": labels}, directory / f"{split}.pt"
        )


def test_loader_supports_dense_and_global_bundles(tmp_path) -> None:
    _write_dense_group(tmp_path)
    _write_global_group(tmp_path)
    dense = load_bundle_split(
        bundle_paths(tmp_path, dataset="example", model="model")["test"]
    )
    global_split = load_bundle_split(
        bundle_paths(tmp_path, dataset="scenes", model="model")["test"]
    )
    assert dense.tensor_layout == "dense_grid_labels"
    assert global_split.tensor_layout == "sample_vectors"


def test_dense_and_global_feature_gathering(tmp_path) -> None:
    _write_dense_group(tmp_path)
    _write_global_group(tmp_path)
    dense = load_bundle_split(
        bundle_paths(tmp_path, dataset="example", model="model")["test"]
    )
    global_split = load_bundle_split(
        bundle_paths(tmp_path, dataset="scenes", model="model")["test"]
    )
    dense_features, dense_unit = gather_features(dense, torch.arange(16).numpy())
    global_features, global_unit = gather_features(
        global_split, torch.arange(16).numpy()
    )
    assert dense_features.shape == (16, 4)
    assert global_features.shape == (16, 4)
    assert dense_unit == "pixel"
    assert global_unit == "sample"


def test_three_diagnostics_use_independent_reports(tmp_path) -> None:
    _write_global_group(tmp_path)
    output = tmp_path / "results"
    kmeans = run_kmeans(
        root=tmp_path,
        dataset="scenes",
        model="model",
        output_dir=output / "kmeans",
        per_class=5,
        n_init=2,
        max_iter=20,
    )
    dbscan = run_dbscan(
        root=tmp_path,
        dataset="scenes",
        model="model",
        output_dir=output / "dbscan",
        per_class=5,
        min_samples=[2],
        eps_multipliers=[1.0],
    )
    retrieval = run_cosine_retrieval(
        root=tmp_path,
        dataset="scenes",
        model="model",
        output_dir=output / "cosine_retrieval",
        gallery_per_class=5,
        query_per_class=3,
        k_values=[1, 5],
    )
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (kmeans, dbscan, retrieval)
    ]
    assert [report["task"] for report in reports] == [
        "kmeans",
        "dbscan",
        "cosine_retrieval",
    ]
    assert all(report["dataset"] == "scenes" for report in reports)
    assert all(report["seed"] == 42 for report in reports)


def test_knn_has_its_own_paper_style_report(tmp_path) -> None:
    _write_global_group(tmp_path)
    (tmp_path / "scenes" / "model" / "test.pt").unlink()
    report_path = run_knn(
        root=tmp_path,
        dataset="scenes",
        model="model",
        output_dir=tmp_path / "results" / "knn",
        split_name="valid",
        k=2,
        batch_size=4,
        device="cpu",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["task"] == "knn"
    assert report["config"]["metric"] == "cosine"
    assert report["config"]["vote"] == "softmax_weighted"
    assert report["split"] == "valid"
    assert set(report["input_audit"]) == {"train", "valid"}


def test_linear_report_is_dataset_independent(tmp_path, monkeypatch) -> None:
    _write_dense_group(tmp_path)

    def fake_train_probe(**kwargs):
        lr = kwargs["lr"]
        return {
            "lr": lr,
            "seed": 42,
            "epochs": kwargs["epochs"],
            "eval_interval": kwargs["eval_interval"],
            "batch_size": kwargs["batch_size"],
            "sample_limit": kwargs["sample_limit"],
            "best_epoch": 1,
            "best_valid_miou": 1.0 - abs(lr - 0.2),
            "evaluation_miou": 0.5,
            "elapsed_seconds": 0.01,
        }

    monkeypatch.setattr(
        "projects.geofm_embeddings.evaluation.linear.train_probe",
        fake_train_probe,
    )
    report_path = run_linear(
        root=tmp_path,
        dataset="example",
        model="model",
        output_dir=tmp_path / "results" / "linear",
        learning_rates=[0.1, 0.2, 0.3],
        device="cpu",
        epochs=1,
        eval_interval=1,
        batch_size=1,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["task"] == "linear"
    assert report["dataset"] == "example"
    assert report["model"] == "model"
    assert report["num_classes"] == 4
    assert report["selected_result"]["lr"] == 0.2


def test_linear_script_routes_sample_labels_to_classification(
    tmp_path, monkeypatch
) -> None:
    _write_global_group(tmp_path)
    (tmp_path / "scenes" / "model" / "test.pt").unlink()

    def fake_train_probe(**kwargs):
        return {
            "lr": kwargs["lr"],
            "best_valid_accuracy": 0.75,
            "evaluation_accuracy": 0.7,
        }

    monkeypatch.setattr(
        "projects.geofm_embeddings.evaluation.linear.train_probe",
        fake_train_probe,
    )
    report_path = run_linear(
        root=tmp_path,
        dataset="scenes",
        model="model",
        output_dir=tmp_path / "results" / "linear_classification",
        learning_rates=[0.001],
        evaluation_split="valid",
        device="cpu",
        epochs=1,
        eval_interval=1,
        batch_size=1,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["linear_mode"] == "classification"
    assert report["evaluation_split"] == "valid"
    assert report["evaluation_is_independent_test"] is False
    assert report["selection_uses_evaluation_split"] is True
    assert report["evaluation_metric"] == "accuracy"
    assert report["evaluation_score"] == 0.7
