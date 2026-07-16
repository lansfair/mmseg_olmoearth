from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .clustering import evaluate_dbscan, evaluate_kmeans
from .intrinsic_dimension import evaluate_id
from .io import (
    align_embedding,
    append_rows,
    embedding_audit,
    load_embedding,
    load_metadata,
)
from .preprocessing import apply_track, l2
from .retrieval import evaluate_semantic_retrieval
from .sampling import balanced_per_split_indices
from .splits import ensure_split
from .supervised import evaluate_knn, evaluate_linear_probe
from .visualization import plot_pca_diagnostics


def _progress(message: str) -> None:
    print(message, flush=True)


def _tag(rows: list[dict[str, Any]], model: str, track: str) -> list[dict[str, Any]]:
    return [{"model": model, "track": track, **row} for row in rows]


def _save_status(path: Path, status: dict[str, Any]) -> None:
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary(run_dir: Path) -> None:
    pieces = []
    for filename, metric, filters in [
        ("clustering.csv", "ari", {"algorithm": "kmeans"}),
        ("knn.csv", "macro_f1", {"k": 5, "budget_per_class": "all"}),
        ("linear_probe.csv", "macro_f1", {"budget_per_class": "all", "status": "ok"}),
        ("retrieval.csv", "mrr", {}),
        ("intrinsic_dimension.csv", "fishers_id", {}),
    ]:
        path = run_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for column, value in filters.items():
            if column in frame:
                frame = frame[frame[column].astype(str) == str(value)]
        if metric not in frame:
            continue
        grouped = frame.groupby(["model", "track"])[metric].agg(["mean", "std", "count"]).reset_index()
        prefix = f"{filename}_{metric}"
        grouped[f"{prefix}_ci95"] = np.where(
            grouped["count"] > 1,
            1.96 * grouped["std"] / np.sqrt(grouped["count"]),
            np.nan,
        )
        grouped = grouped.rename(columns={
            "mean": f"{prefix}_mean",
            "std": f"{prefix}_std",
            "count": f"{prefix}_n",
        })
        pieces.append(grouped)
    if not pieces:
        return
    summary = pieces[0]
    for piece in pieces[1:]:
        summary = summary.merge(piece, on=["model", "track"], how="outer")
    summary.to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")

    id_col = "intrinsic_dimension.csv_fishers_id_mean"
    linear_col = "linear_probe.csv_macro_f1_mean"
    valid = summary[[id_col, linear_col]].dropna() if {id_col, linear_col} <= set(summary) else pd.DataFrame()
    if len(valid) >= 3:
        rho, p_value = spearmanr(valid[id_col], valid[linear_col])
        pd.DataFrame([{"x": id_col, "y": linear_col, "spearman_rho": rho, "p_value": p_value, "n": len(valid)}]).to_csv(
            run_dir / "id_performance_correlation.csv", index=False, encoding="utf-8-sig"
        )


_ACTIVE_STATUS_PATH: Path | None = None


def run_experiment(config_path: str | Path) -> Path:
    global _ACTIVE_STATUS_PATH
    try:
        return _run_experiment_impl(config_path)
    except Exception as error:
        if _ACTIVE_STATUS_PATH is not None and _ACTIVE_STATUS_PATH.exists():
            status = json.loads(_ACTIVE_STATUS_PATH.read_text(encoding="utf-8"))
            status.update({
                "state": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            })
            _save_status(_ACTIVE_STATUS_PATH, status)
        raise
    finally:
        _ACTIVE_STATUS_PATH = None


def _run_experiment_impl(config_path: str | Path) -> Path:
    global _ACTIVE_STATUS_PATH
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent
    experiment_id = config.get("experiment_id", "embedding_evaluation")
    root = Path(config.get("output_dir", "results"))
    if not root.is_absolute():
        root = (base_dir / root).resolve()
    run_dir = root / experiment_id / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata_spec = config["metadata"]
    metadata = load_metadata(metadata_spec, base_dir)
    seed = int(config.get("random_seed", 42))
    metadata = ensure_split(metadata, metadata_spec, config.get("split", {}), seed)
    metadata.to_csv(run_dir / "split_assignment.csv", index=False, encoding="utf-8-sig")
    label_col = metadata_spec.get("label_column", "label")
    id_col = metadata_spec.get("id_column", "sample_id")
    if label_col not in metadata:
        raise ValueError(f"Metadata is missing label column: {label_col}")
    split_counts = (
        metadata.groupby(["_split", label_col], dropna=False)
        .size()
        .rename("samples")
        .reset_index()
    )
    split_counts.to_csv(
        run_dir / "split_class_counts.csv", index=False, encoding="utf-8-sig"
    )

    seeds = [int(value) for value in config.get("seeds", [42, 43, 44, 45, 46])]
    if not seeds:
        raise ValueError("seeds must contain at least one integer")
    tasks = config.get("tasks", {})
    models = config["models"]
    sample_policy = config.get("sample_policy", "strict")
    if sample_policy not in {"strict", "per_model"}:
        raise ValueError("sample_policy must be 'strict' or 'per_model'")
    nonfinite_policy = config.get("nonfinite_policy", "error")
    status: dict[str, Any] = {"state": "running", "completed": [], "run_dir": str(run_dir)}
    _ACTIVE_STATUS_PATH = run_dir / "status.json"
    _save_status(_ACTIVE_STATUS_PATH, status)
    reference_ids: np.ndarray | None = None

    for model_index, model_spec in enumerate(models, start=1):
        model_name = model_spec["name"]
        _progress(f"[{model_index}/{len(models)}] Loading model: {model_name}")
        embedding = load_embedding(model_spec, base_dir)
        append_rows(
            run_dir / "input_audit.csv",
            [embedding_audit(embedding, metadata, id_col)],
        )
        values, aligned_meta = align_embedding(
            embedding, metadata, id_col, nonfinite_policy=nonfinite_policy
        )
        aligned_ids = aligned_meta[id_col].to_numpy()
        if reference_ids is None:
            reference_ids = aligned_ids
        elif sample_policy == "strict" and not np.array_equal(reference_ids, aligned_ids):
            raise ValueError(
                f"{model_name}: aligned sample IDs differ from the first model. "
                "Use the exact common sample set for a fair comparison."
            )
        labels = aligned_meta[label_col].astype(str).to_numpy()
        split = aligned_meta["_split"].to_numpy()
        sampling_cfg = config.get("sampling", {})
        per_class_limit = sampling_cfg.get("max_samples_per_class_per_split")
        if per_class_limit is not None:
            selected = balanced_per_split_indices(
                labels, split, int(per_class_limit), seed
            )
            values = values[selected]
            aligned_meta = aligned_meta.iloc[selected].reset_index(drop=True)
            labels = labels[selected]
            split = split[selected]
        evaluated_counts = (
            pd.DataFrame({"split": split, "label": labels})
            .groupby(["split", "label"]).size().rename("samples").reset_index()
        )
        evaluated_counts.insert(0, "model", model_name)
        append_rows(run_dir / "evaluated_class_counts.csv", evaluated_counts.to_dict("records"))
        train_mask = split == "train"
        if not train_mask.any() or not np.any(split == "test"):
            raise ValueError(f"{model_name}: aligned data has no train or test samples")

        tracks = config.get("tracks", [{"name": "native", "pca_dim": None}])
        for track_index, track_spec in enumerate(tracks, start=1):
            track_name = track_spec["name"]
            _progress(f"  [{track_index}/{len(tracks)}] Track: {track_name}")
            tracked = apply_track(values, train_mask, track_spec.get("pca_dim"), seed)
            normalized = l2(tracked)

            if tasks.get("clustering", {}).get("enabled", True):
                _progress("    K-means and DBSCAN")
                clustering_cfg = tasks["clustering"]
                max_samples = min(int(clustering_cfg.get("max_samples", 50000)), len(normalized))
                rng = np.random.default_rng(seed)
                selected = np.sort(rng.choice(len(normalized), max_samples, replace=False))
                cluster_x, cluster_y = normalized[selected], labels[selected]
                rows = evaluate_kmeans(cluster_x, cluster_y, clustering_cfg.get("kmeans", {}), seeds)
                dbscan_max = min(
                    int(clustering_cfg.get("dbscan_max_samples", 10000)), max_samples
                )
                dbscan_selected = np.sort(rng.choice(len(normalized), dbscan_max, replace=False))
                dbscan_rows = evaluate_dbscan(
                    normalized[dbscan_selected],
                    labels[dbscan_selected],
                    clustering_cfg.get("dbscan", {}),
                    seed,
                )
                for row in dbscan_rows:
                    row["n_samples"] = dbscan_max
                rows += dbscan_rows
                for row in rows:
                    row.setdefault("n_samples", max_samples)
                append_rows(run_dir / "clustering.csv", _tag(rows, model_name, track_name))

            if tasks.get("knn", {}).get("enabled", True):
                _progress("    KNN probe")
                rows = evaluate_knn(normalized, labels, split, tasks["knn"], seeds)
                append_rows(run_dir / "knn.csv", _tag(rows, model_name, track_name))

            if tasks.get("linear_probe", {}).get("enabled", True):
                _progress("    Linear probe")
                rows = evaluate_linear_probe(tracked, labels, split, tasks["linear_probe"], seeds)
                append_rows(run_dir / "linear_probe.csv", _tag(rows, model_name, track_name))

            retrieval_cfg = tasks.get("retrieval", {})
            if retrieval_cfg.get("enabled", False):
                _progress("    Semantic retrieval")
                rows, details = evaluate_semantic_retrieval(
                    normalized,
                    labels,
                    aligned_meta[id_col].astype(str).to_numpy(),
                    split,
                    retrieval_cfg,
                )
                append_rows(run_dir / "retrieval.csv", _tag(rows, model_name, track_name))
                safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", f"{model_name}_{track_name}")
                pd.DataFrame(details).to_csv(
                    run_dir / f"retrieval_details_{safe_name}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            id_cfg = tasks.get("intrinsic_dimension", {})
            if id_cfg.get("enabled", True) and track_name in id_cfg.get("tracks", ["native"]):
                _progress("    Intrinsic dimension")
                id_seeds = seeds[: int(id_cfg.get("repeats", 3))]
                rows, local_id = evaluate_id(tracked, id_cfg, id_seeds)
                append_rows(run_dir / "intrinsic_dimension.csv", _tag(rows, model_name, track_name))
                local = aligned_meta[[id_col]].copy()
                for source, output in [
                    (metadata_spec.get("latitude_column", "latitude"), "latitude"),
                    (metadata_spec.get("longitude_column", "longitude"), "longitude"),
                ]:
                    if source in aligned_meta:
                        local[output] = aligned_meta[source]
                local["local_mle_id"] = local_id
                local.to_csv(run_dir / f"local_id_{model_name}_{track_name}.csv", index=False, encoding="utf-8-sig")

            visualization_cfg = tasks.get("visualization", {})
            if visualization_cfg.get("enabled", False) and track_name in visualization_cfg.get(
                "tracks", [track_name]
            ):
                _progress("    PCA diagnostics")
                safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", f"{model_name}_{track_name}")
                plot_pca_diagnostics(
                    tracked,
                    labels,
                    split,
                    aligned_meta[id_col].astype(str).to_numpy(),
                    run_dir / "figures" / f"pca_{safe_name}.png",
                    visualization_cfg,
                    seed,
                )

            completed = f"{model_name}/{track_name}"
            status["completed"].append(completed)
            _save_status(run_dir / "status.json", status)

    _write_summary(run_dir)
    status["state"] = "complete"
    _save_status(run_dir / "status.json", status)
    _progress(f"Complete. Results: {run_dir}")
    return run_dir
