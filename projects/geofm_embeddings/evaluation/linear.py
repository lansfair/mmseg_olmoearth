"""Dataset-independent linear probing for classification and segmentation."""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .bundle import (
    BUNDLE_INPUT_FORMAT,
    FIXED_SEED,
    BundleSplit,
    bundle_paths,
    load_bundle_split,
    write_json_atomic,
)


LINEAR_PROTOCOL = "olmoearth_linear_probe_fixed_seed_lr_sweep_v1"
TRAIN_ONLY_PROTOCOL = "geofm_linear_probe_train_only_fixed_seed_v1"


class BundleDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, split: BundleSplit, limit: int | None = None) -> None:
        self.split = split
        self.length = len(split.embeddings)
        if limit is not None:
            if limit < 1:
                raise ValueError("sample_limit must be positive")
            self.length = min(self.length, limit)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.split.embeddings[index], self.split.labels[index]


class ClassificationBundleDataset(BundleDataset):
    """Return one vector per sample, pooling a spatial grid when necessary."""

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        embedding = self.split.embeddings[index]
        if embedding.ndim == 3:
            embedding = embedding.float().mean(dim=(0, 1))
        return embedding, self.split.labels[index]


class DenseLinearProbe(nn.Module):
    """One affine classifier per native feature token, then bilinear resize."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, embeddings: Tensor, label_size: tuple[int, int]) -> Tensor:
        if embeddings.ndim != 4:
            raise ValueError(
                f"Dense embeddings must be [B,H,W,D], got {embeddings.shape}"
            )
        logits = self.linear(embeddings).permute(0, 3, 1, 2)
        if tuple(logits.shape[-2:]) != tuple(label_size):
            logits = F.interpolate(
                logits, size=label_size, mode="bilinear", align_corners=False
            )
        return logits


class ClassificationLinearProbe(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim != 2:
            raise ValueError(
                f"Classification embeddings must be [B,D], got {embeddings.shape}"
            )
        return self.linear(embeddings)


def _save_probe_checkpoint(
    path: Path,
    probe: nn.Module,
    *,
    mode: str,
    in_features: int,
    class_values: tuple[int, ...],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "geofm_linear_probe_v1",
            "mode": mode,
            "in_features": in_features,
            "class_values": torch.tensor(class_values, dtype=torch.int64),
            "state_dict": {
                key: value.detach().cpu() for key, value in probe.state_dict().items()
            },
        },
        temporary,
    )
    temporary.replace(path)
    return path


def _label_values(labels: Tensor, ignore_label: int) -> set[int]:
    result: set[int] = set()
    for start in range(0, len(labels), 64):
        values = torch.unique(labels[start : start + 64]).tolist()
        result.update(
            int(value)
            for value in values
            if int(value) != ignore_label and int(value) >= 0
        )
    return result


def infer_class_values(
    train: BundleSplit,
    valid: BundleSplit,
    evaluation: BundleSplit,
    *,
    ignore_label: int,
) -> tuple[int, ...]:
    train_values = _label_values(train.labels, ignore_label)
    if len(train_values) < 2:
        raise ValueError("Training labels must contain at least two classes")
    for name, split in (("valid", valid), ("evaluation", evaluation)):
        unknown = _label_values(split.labels, ignore_label).difference(train_values)
        if unknown:
            raise ValueError(
                f"{name} contains classes absent from train: {sorted(unknown)}"
            )
    return tuple(sorted(train_values))


def remap_labels(
    labels: Tensor, class_values: tuple[int, ...], ignore_label: int
) -> Tensor:
    mapped = torch.full_like(labels, -1, dtype=torch.int64)
    for contiguous, original in enumerate(class_values):
        mapped[labels == original] = contiguous
    mapped[(labels == ignore_label) | (labels < 0)] = -1
    return mapped


def _loader(
    split: BundleSplit,
    *,
    batch_size: int,
    shuffle: bool,
    sample_limit: int | None,
    num_workers: int,
    pin_memory: bool,
    classification: bool = False,
) -> DataLoader:
    dataset_class = ClassificationBundleDataset if classification else BundleDataset
    return DataLoader(
        dataset_class(split, sample_limit),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(FIXED_SEED),
        pin_memory=pin_memory,
    )


def _adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: float,
    *,
    max_lr: float,
    total_epochs: int,
) -> float:
    warmup_epochs = int(total_epochs * 0.1)
    min_lr = 1.0e-5
    if warmup_epochs and epoch < warmup_epochs:
        lr = max_lr * epoch / warmup_epochs
    else:
        denominator = max(total_epochs - warmup_epochs, 1)
        lr = min_lr + (max_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - warmup_epochs) / denominator)
        )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _confusion_update(
    confusion: Tensor,
    predictions: Tensor,
    labels: Tensor,
    *,
    num_classes: int,
) -> None:
    predictions = predictions.reshape(-1).to(torch.int64)
    labels = labels.reshape(-1).to(torch.int64)
    valid = (labels >= 0) & (labels < num_classes)
    encoded = labels[valid] * num_classes + predictions[valid]
    encoded = encoded.to(confusion.device, non_blocking=True)
    confusion += torch.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def mean_iou_from_confusion(confusion: Tensor) -> tuple[float, list[float | None]]:
    confusion = confusion.to(torch.float64)
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    valid = union > 0
    per_class = torch.where(valid, intersection / union, torch.nan)
    mean = float(per_class[valid].mean()) if valid.any() else float("nan")
    return mean, [None if torch.isnan(value) else float(value) for value in per_class]


def evaluate_probe(
    probe: DenseLinearProbe,
    loader: DataLoader,
    *,
    device: torch.device,
    class_values: tuple[int, ...],
    ignore_label: int,
) -> dict[str, Any]:
    probe.eval()
    confusion = torch.zeros(
        (probe.num_classes, probe.num_classes), dtype=torch.int64, device=device
    )
    nonfinite_batches = 0
    with torch.inference_mode():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device, non_blocking=device.type == "cuda")
            if device.type != "cuda":
                embeddings = embeddings.float()
            labels = remap_labels(labels, class_values, ignore_label).to(
                device, non_blocking=device.type == "cuda"
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = probe(embeddings, tuple(labels.shape[-2:]))
            nonfinite_batches += int((~torch.isfinite(logits)).any())
            _confusion_update(
                confusion, logits.argmax(dim=1), labels, num_classes=probe.num_classes
            )
    confusion = confusion.cpu()
    mean_iou, class_iou = mean_iou_from_confusion(confusion)
    return {
        "miou": mean_iou,
        "class_iou": class_iou,
        "confusion_matrix": confusion.tolist(),
        "nonfinite_batches": nonfinite_batches,
    }


def classification_metrics_from_confusion(confusion: Tensor) -> dict[str, Any]:
    values = confusion.to(torch.float64)
    total = values.sum()
    if total <= 0:
        raise ValueError("Classification evaluation has no non-ignored samples")
    true_count = values.sum(dim=1)
    predicted_count = values.sum(dim=0)
    true_positive = values.diag()
    recall = torch.where(true_count > 0, true_positive / true_count, torch.nan)
    precision = torch.where(
        predicted_count > 0, true_positive / predicted_count, torch.nan
    )
    denominator = precision + recall
    f1 = torch.where(denominator > 0, 2 * precision * recall / denominator, 0.0)
    valid_classes = true_count > 0
    return {
        "accuracy": float(true_positive.sum() / total),
        "balanced_accuracy": float(recall[valid_classes].mean()),
        "macro_f1": float(f1[valid_classes].mean()),
        "class_recall": [
            None if torch.isnan(value) else float(value) for value in recall
        ],
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_classification_probe(
    probe: ClassificationLinearProbe,
    loader: DataLoader,
    *,
    device: torch.device,
    class_values: tuple[int, ...],
    ignore_label: int,
) -> dict[str, Any]:
    probe.eval()
    confusion = torch.zeros(
        (probe.num_classes, probe.num_classes), dtype=torch.int64, device=device
    )
    nonfinite_batches = 0
    with torch.inference_mode():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device, non_blocking=device.type == "cuda")
            if device.type != "cuda":
                embeddings = embeddings.float()
            labels = remap_labels(labels, class_values, ignore_label).to(
                device, non_blocking=device.type == "cuda"
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = probe(embeddings)
            nonfinite_batches += int((~torch.isfinite(logits)).any())
            _confusion_update(
                confusion, logits.argmax(dim=1), labels, num_classes=probe.num_classes
            )
    result = classification_metrics_from_confusion(confusion.cpu())
    result["nonfinite_batches"] = nonfinite_batches
    return result


def _train_dense_probe(
    *,
    train: BundleSplit,
    valid: BundleSplit,
    evaluation: BundleSplit,
    class_values: tuple[int, ...],
    ignore_label: int,
    lr: float,
    device: str = "cuda",
    epochs: int = 50,
    eval_interval: int = 5,
    batch_size: int = 8,
    num_workers: int = 0,
    sample_limit: int | None = None,
    checkpoint_path: Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if epochs < 1 or eval_interval < 1 or batch_size < 1:
        raise ValueError("epochs, eval_interval, and batch_size must be positive")
    for name, split in (
        ("train", train),
        ("valid", valid),
        ("evaluation", evaluation),
    ):
        if split.tensor_layout != "dense_grid_labels":
            raise ValueError(
                f"Dense linear requires [N,H,W,D] embeddings and [N,H,W] labels; "
                f"{name} is {split.tensor_layout}"
            )
    dimensions = {
        int(split.embeddings.shape[-1]) for split in (train, valid, evaluation)
    }
    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimensions differ across splits: {dimensions}")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    torch.manual_seed(FIXED_SEED)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(FIXED_SEED)
    in_features = dimensions.pop()
    probe = DenseLinearProbe(in_features, len(class_values)).to(torch_device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    train_loader = _loader(
        train,
        batch_size=batch_size,
        shuffle=True,
        sample_limit=sample_limit,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )
    valid_loader = _loader(
        valid,
        batch_size=batch_size,
        shuffle=False,
        sample_limit=sample_limit,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    history: list[dict[str, Any]] = []
    best_state: dict[str, Tensor] | None = None
    best_valid_miou = -math.inf
    best_epoch = 0
    started = time.perf_counter()
    for epoch_index in range(epochs):
        probe.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_index, (embeddings, labels) in enumerate(train_loader):
            embeddings = embeddings.to(
                torch_device, non_blocking=torch_device.type == "cuda"
            )
            if torch_device.type != "cuda":
                embeddings = embeddings.float()
            labels = remap_labels(labels, class_values, ignore_label).to(
                torch_device, non_blocking=torch_device.type == "cuda"
            )
            with torch.autocast(
                device_type=torch_device.type,
                dtype=torch.bfloat16,
                enabled=torch_device.type == "cuda",
            ):
                logits = probe(embeddings, tuple(labels.shape[-2:]))
                loss = loss_function(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch_index + 1}, batch {batch_index}"
                )
            loss.backward()
            current_lr = _adjust_learning_rate(
                optimizer,
                epoch_index + batch_index / len(train_loader),
                max_lr=lr,
                total_epochs=epochs,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            count = len(labels)
            loss_sum += float(loss.detach()) * count
            sample_count += count
        epoch = epoch_index + 1
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            "lr": current_lr,
        }
        if epoch % eval_interval == 0 or epoch == epochs:
            validation = evaluate_probe(
                probe,
                valid_loader,
                device=torch_device,
                class_values=class_values,
                ignore_label=ignore_label,
            )
            row["valid_miou"] = validation["miou"]
            row["valid_nonfinite_batches"] = validation["nonfinite_batches"]
            if validation["miou"] > best_valid_miou:
                best_valid_miou = validation["miou"]
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in probe.state_dict().items()
                }
        history.append(row)
        if progress_callback is not None:
            progress_callback(dict(row))
    if best_state is None:
        raise RuntimeError("Validation did not produce a finite best checkpoint")
    probe.load_state_dict(best_state)
    saved_checkpoint = None
    if checkpoint_path is not None:
        saved_checkpoint = _save_probe_checkpoint(
            checkpoint_path,
            probe,
            mode="dense_segmentation",
            in_features=in_features,
            class_values=class_values,
        )
    evaluation_loader = _loader(
        evaluation,
        batch_size=batch_size,
        shuffle=False,
        sample_limit=sample_limit,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )
    evaluation_result = evaluate_probe(
        probe,
        evaluation_loader,
        device=torch_device,
        class_values=class_values,
        ignore_label=ignore_label,
    )
    return {
        "lr": lr,
        "seed": FIXED_SEED,
        "epochs": epochs,
        "eval_interval": eval_interval,
        "batch_size": batch_size,
        "sample_limit": sample_limit,
        "best_epoch": best_epoch,
        "best_valid_miou": best_valid_miou,
        "evaluation_miou": evaluation_result["miou"],
        "evaluation_class_iou": evaluation_result["class_iou"],
        "evaluation_confusion_matrix": evaluation_result["confusion_matrix"],
        "evaluation_nonfinite_batches": evaluation_result["nonfinite_batches"],
        "checkpoint_path": (
            str(saved_checkpoint) if saved_checkpoint is not None else None
        ),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _train_classification_probe(
    *,
    train: BundleSplit,
    valid: BundleSplit,
    evaluation: BundleSplit,
    class_values: tuple[int, ...],
    ignore_label: int,
    lr: float,
    device: str = "cuda",
    epochs: int = 50,
    eval_interval: int = 5,
    batch_size: int = 8,
    num_workers: int = 0,
    sample_limit: int | None = None,
    checkpoint_path: Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if epochs < 1 or eval_interval < 1 or batch_size < 1:
        raise ValueError("epochs, eval_interval, and batch_size must be positive")
    dimensions = {
        int(split.embeddings.shape[-1]) for split in (train, valid, evaluation)
    }
    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimensions differ across splits: {dimensions}")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    torch.manual_seed(FIXED_SEED)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(FIXED_SEED)
    in_features = dimensions.pop()
    probe = ClassificationLinearProbe(in_features, len(class_values)).to(
        torch_device
    )
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    train_loader = _loader(
        train,
        batch_size=batch_size,
        shuffle=True,
        sample_limit=sample_limit,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
        classification=True,
    )
    valid_loader = _loader(
        valid,
        batch_size=batch_size,
        shuffle=False,
        sample_limit=sample_limit,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
        classification=True,
    )
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    history: list[dict[str, Any]] = []
    best_state: dict[str, Tensor] | None = None
    best_valid_accuracy = -math.inf
    best_epoch = 0
    started = time.perf_counter()
    for epoch_index in range(epochs):
        probe.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_index, (embeddings, labels) in enumerate(train_loader):
            embeddings = embeddings.to(
                torch_device, non_blocking=torch_device.type == "cuda"
            )
            if torch_device.type != "cuda":
                embeddings = embeddings.float()
            labels = remap_labels(labels, class_values, ignore_label).to(
                torch_device, non_blocking=torch_device.type == "cuda"
            )
            with torch.autocast(
                device_type=torch_device.type,
                dtype=torch.bfloat16,
                enabled=torch_device.type == "cuda",
            ):
                logits = probe(embeddings)
                loss = loss_function(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch_index + 1}, batch {batch_index}"
                )
            loss.backward()
            current_lr = _adjust_learning_rate(
                optimizer,
                epoch_index + batch_index / len(train_loader),
                max_lr=lr,
                total_epochs=epochs,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            count = len(labels)
            loss_sum += float(loss.detach()) * count
            sample_count += count
        epoch = epoch_index + 1
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            "lr": current_lr,
        }
        if epoch % eval_interval == 0 or epoch == epochs:
            validation = evaluate_classification_probe(
                probe,
                valid_loader,
                device=torch_device,
                class_values=class_values,
                ignore_label=ignore_label,
            )
            row["valid_accuracy"] = validation["accuracy"]
            row["valid_macro_f1"] = validation["macro_f1"]
            row["valid_nonfinite_batches"] = validation["nonfinite_batches"]
            if validation["accuracy"] > best_valid_accuracy:
                best_valid_accuracy = validation["accuracy"]
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in probe.state_dict().items()
                }
        history.append(row)
        if progress_callback is not None:
            progress_callback(dict(row))
    if best_state is None:
        raise RuntimeError("Validation did not produce a finite best checkpoint")
    probe.load_state_dict(best_state)
    saved_checkpoint = None
    if checkpoint_path is not None:
        saved_checkpoint = _save_probe_checkpoint(
            checkpoint_path,
            probe,
            mode="classification",
            in_features=in_features,
            class_values=class_values,
        )
    evaluation_loader = _loader(
        evaluation,
        batch_size=batch_size,
        shuffle=False,
        sample_limit=sample_limit,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
        classification=True,
    )
    evaluation_result = evaluate_classification_probe(
        probe,
        evaluation_loader,
        device=torch_device,
        class_values=class_values,
        ignore_label=ignore_label,
    )
    return {
        "lr": lr,
        "seed": FIXED_SEED,
        "epochs": epochs,
        "eval_interval": eval_interval,
        "batch_size": batch_size,
        "sample_limit": sample_limit,
        "best_epoch": best_epoch,
        "best_valid_accuracy": best_valid_accuracy,
        "evaluation_accuracy": evaluation_result["accuracy"],
        "evaluation_balanced_accuracy": evaluation_result["balanced_accuracy"],
        "evaluation_macro_f1": evaluation_result["macro_f1"],
        "evaluation_class_recall": evaluation_result["class_recall"],
        "evaluation_confusion_matrix": evaluation_result["confusion_matrix"],
        "evaluation_nonfinite_batches": evaluation_result["nonfinite_batches"],
        "checkpoint_path": (
            str(saved_checkpoint) if saved_checkpoint is not None else None
        ),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }


def train_probe(
    *,
    train: BundleSplit,
    valid: BundleSplit,
    evaluation: BundleSplit,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a linear classifier or dense linear head for the selected script."""

    splits = (train, valid, evaluation)
    layouts = {split.tensor_layout for split in splits}
    if len(layouts) != 1:
        raise ValueError(
            f"Bundle tensor layouts differ across splits: {sorted(layouts)}"
        )
    common = {
        "train": train,
        "valid": valid,
        "evaluation": evaluation,
        **kwargs,
    }
    if layouts == {"dense_grid_labels"}:
        return _train_dense_probe(**common)
    if all(split.labels.ndim == 1 for split in splits):
        return _train_classification_probe(**common)
    raise ValueError(f"Unsupported linear-probe tensor layout: {sorted(layouts)}")


def train_only_probe(
    *,
    train: BundleSplit,
    class_values: tuple[int, ...],
    ignore_label: int,
    lr: float,
    checkpoint_path: Path,
    device: str = "cuda",
    epochs: int = 50,
    batch_size: int = 8,
    num_workers: int = 0,
    sample_limit: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit one fixed-hyperparameter probe on all labeled training samples."""

    if lr <= 0 or epochs < 1 or batch_size < 1:
        raise ValueError("lr, epochs, and batch_size must be positive")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    torch.manual_seed(FIXED_SEED)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(FIXED_SEED)

    in_features = int(train.embeddings.shape[-1])
    if train.tensor_layout == "dense_grid_labels":
        mode = "dense_segmentation"
        probe: DenseLinearProbe | ClassificationLinearProbe = DenseLinearProbe(
            in_features, len(class_values)
        )
        classification = False
    elif train.labels.ndim == 1:
        mode = "classification"
        probe = ClassificationLinearProbe(in_features, len(class_values))
        classification = True
    else:
        raise ValueError(
            f"Unsupported train-only tensor layout: {train.tensor_layout}"
        )
    probe = probe.to(torch_device)
    loader = _loader(
        train,
        batch_size=batch_size,
        shuffle=True,
        sample_limit=sample_limit,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
        classification=classification,
    )
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch_index in range(epochs):
        probe.train()
        loss_sum = 0.0
        item_count = 0
        for batch_index, (embeddings, labels) in enumerate(loader):
            embeddings = embeddings.to(
                torch_device, non_blocking=torch_device.type == "cuda"
            )
            if torch_device.type != "cuda":
                embeddings = embeddings.float()
            labels = remap_labels(labels, class_values, ignore_label).to(
                torch_device, non_blocking=torch_device.type == "cuda"
            )
            with torch.autocast(
                device_type=torch_device.type,
                dtype=torch.bfloat16,
                enabled=torch_device.type == "cuda",
            ):
                if mode == "dense_segmentation":
                    logits = probe(embeddings, tuple(labels.shape[-2:]))
                else:
                    logits = probe(embeddings)
                loss = loss_function(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch_index + 1}, "
                    f"batch {batch_index}"
                )
            loss.backward()
            current_lr = _adjust_learning_rate(
                optimizer,
                epoch_index + batch_index / len(loader),
                max_lr=lr,
                total_epochs=epochs,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            count = len(labels)
            loss_sum += float(loss.detach()) * count
            item_count += count
        row = {
            "epoch": epoch_index + 1,
            "train_loss": loss_sum / item_count,
            "lr": current_lr,
        }
        history.append(row)
        if progress_callback is not None:
            progress_callback(dict(row))

    checkpoint = _save_probe_checkpoint(
        checkpoint_path,
        probe,
        mode=mode,
        in_features=in_features,
        class_values=class_values,
    )
    return {
        "mode": mode,
        "seed": FIXED_SEED,
        "max_learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "sample_limit": sample_limit,
        "checkpoint_path": str(checkpoint),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_linear_train_only(
    *,
    root: str | Path,
    dataset: str,
    model: str,
    output_dir: str | Path,
    learning_rate: float,
    ignore_label: int = -1,
    device: str = "cuda",
    epochs: int = 50,
    batch_size: int = 8,
    num_workers: int = 0,
    sample_limit: int | None = None,
) -> Path:
    """Train on all of train.pt without validation or model selection."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    checkpoint_path = output_dir / "linear_probe.pth"
    status: dict[str, Any] = {
        "state": "running",
        "task": "linear_train_only",
        "dataset": dataset,
        "model": model,
    }
    write_json_atomic(status_path, status)
    try:
        train_path = bundle_paths(
            root, dataset=dataset, model=model, splits=("train",)
        )["train"]
        train = load_bundle_split(train_path)
        class_values = tuple(sorted(_label_values(train.labels, ignore_label)))
        if len(class_values) < 2:
            raise ValueError("Training labels must contain at least two classes")

        def progress(row: dict[str, Any]) -> None:
            status["current_epoch"] = row
            write_json_atomic(status_path, status)

        result = train_only_probe(
            train=train,
            class_values=class_values,
            ignore_label=ignore_label,
            lr=float(learning_rate),
            checkpoint_path=checkpoint_path,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            sample_limit=sample_limit,
            progress_callback=progress,
        )
        report = {
            "task": "linear_train_only",
            "protocol": TRAIN_ONLY_PROTOCOL,
            "input_format": BUNDLE_INPUT_FORMAT,
            "dataset": dataset,
            "model": model,
            "seed": FIXED_SEED,
            "ignore_label": ignore_label,
            "class_values": list(class_values),
            "num_classes": len(class_values),
            "selection_metric": None,
            "validation_used": False,
            "input_audit": {"train": train.audit},
            "result": result,
            "checkpoint": str(checkpoint_path),
        }
        report_path = write_json_atomic(output_dir / "report.json", report)
        status.update(
            {
                "state": "complete",
                "current_epoch": None,
                "checkpoint": str(checkpoint_path),
                "report": str(report_path),
            }
        )
        write_json_atomic(status_path, status)
        return report_path
    except Exception as error:
        status.update(
            {
                "state": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json_atomic(status_path, status)
        raise


def run_linear(
    *,
    root: str | Path,
    dataset: str,
    model: str,
    output_dir: str | Path,
    learning_rates: Iterable[float],
    evaluation_split: str = "test",
    ignore_label: int = -1,
    device: str = "cuda",
    epochs: int = 50,
    eval_interval: int = 5,
    batch_size: int = 8,
    num_workers: int = 0,
    sample_limit: int | None = None,
) -> Path:
    if evaluation_split not in {"valid", "test"}:
        raise ValueError("evaluation_split must be 'valid' or 'test'")
    learning_rates = tuple(float(value) for value in learning_rates)
    if not learning_rates or min(learning_rates) <= 0:
        raise ValueError("learning_rates must contain positive values")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    partial_path = output_dir / "runs.partial.json"
    config = {
        "dataset": dataset,
        "model": model,
        "evaluation_split": evaluation_split,
        "learning_rates": list(learning_rates),
        "seed": FIXED_SEED,
        "ignore_label": ignore_label,
        "epochs": epochs,
        "eval_interval": eval_interval,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "sample_limit": sample_limit,
    }
    status: dict[str, Any] = {
        "state": "running",
        "task": "linear",
        "dataset": dataset,
        "model": model,
        "completed_runs": 0,
    }
    write_json_atomic(status_path, status)
    try:
        required_splits = (
            ("train", "valid")
            if evaluation_split == "valid"
            else ("train", "valid", "test")
        )
        paths = bundle_paths(root, dataset=dataset, model=model, splits=required_splits)
        splits = {name: load_bundle_split(path) for name, path in paths.items()}
        evaluation = splits[evaluation_split]
        layouts = {split.tensor_layout for split in splits.values()}
        if layouts == {"dense_grid_labels"}:
            linear_mode = "dense_segmentation"
            selection_key = "best_valid_miou"
            evaluation_key = "evaluation_miou"
        elif len(layouts) == 1 and all(
            split.labels.ndim == 1 for split in splits.values()
        ):
            linear_mode = "classification"
            selection_key = "best_valid_accuracy"
            evaluation_key = "evaluation_accuracy"
        else:
            raise ValueError(
                f"Bundle tensor layouts are incompatible with linear probing: "
                f"{sorted(layouts)}"
            )
        config["linear_mode"] = linear_mode
        class_values = infer_class_values(
            splits["train"],
            splits["valid"],
            evaluation,
            ignore_label=ignore_label,
        )
        results: list[dict[str, Any]] = []
        if partial_path.is_file():
            try:
                partial = json.loads(partial_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                partial = {}
            if partial.get("config") == config:
                results = list(partial.get("runs", []))

        def save_partial() -> None:
            write_json_atomic(partial_path, {"config": config, "runs": results})

        def execute(lr: float) -> dict[str, Any]:
            checkpoint_path = output_dir / "checkpoints" / f"lr_{lr:.12g}.pth"
            reusable = next((row for row in results if float(row["lr"]) == lr), None)
            if reusable is not None and checkpoint_path.is_file():
                return reusable
            if reusable is not None:
                results.remove(reusable)
            status["current_run"] = {"lr": lr, "epoch": 0, "epochs": epochs}
            write_json_atomic(status_path, status)

            def progress(row: dict[str, Any]) -> None:
                status["current_run"] = {"lr": lr, "epochs": epochs, **row}
                write_json_atomic(status_path, status)

            result = train_probe(
                train=splits["train"],
                valid=splits["valid"],
                evaluation=evaluation,
                class_values=class_values,
                ignore_label=ignore_label,
                lr=lr,
                device=device,
                epochs=epochs,
                eval_interval=eval_interval,
                batch_size=batch_size,
                num_workers=num_workers,
                sample_limit=sample_limit,
                checkpoint_path=checkpoint_path,
                progress_callback=progress,
            )
            results.append(result)
            save_partial()
            status["completed_runs"] = len(results)
            write_json_atomic(status_path, status)
            return result

        for lr in learning_rates:
            execute(lr)
        selected = max(results, key=lambda row: row[selection_key])
        selected_checkpoint = Path(selected["checkpoint_path"])
        best_checkpoint = output_dir / "best_probe.pth"
        temporary_checkpoint = best_checkpoint.with_suffix(".pth.tmp")
        shutil.copyfile(selected_checkpoint, temporary_checkpoint)
        temporary_checkpoint.replace(best_checkpoint)
        report = {
            "task": "linear",
            "protocol": LINEAR_PROTOCOL,
            "input_format": BUNDLE_INPUT_FORMAT,
            "dataset": dataset,
            "model": model,
            "seed": FIXED_SEED,
            "ignore_label": ignore_label,
            "class_values": list(class_values),
            "num_classes": len(class_values),
            "linear_mode": linear_mode,
            "selection_metric": selection_key.removeprefix("best_valid_"),
            "evaluation_split": evaluation_split,
            "evaluation_is_independent_test": evaluation_split == "test",
            "selection_uses_evaluation_split": evaluation_split == "valid",
            "evaluation_metric": evaluation_key.removeprefix("evaluation_"),
            "input_audit": {name: split.audit for name, split in splits.items()},
            "config": config,
            "runs": results,
            "selected_result": selected,
            "best_checkpoint": str(best_checkpoint),
            "evaluation_score": float(selected[evaluation_key]),
        }
        report_path = write_json_atomic(output_dir / "report.json", report)
        partial_path.unlink(missing_ok=True)
        status.update(
            {
                "state": "complete",
                "completed_runs": len(results),
                "current_run": None,
                "report": str(report_path),
            }
        )
        write_json_atomic(status_path, status)
        return report_path
    except Exception as error:
        status.update(
            {
                "state": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json_atomic(status_path, status)
        raise
