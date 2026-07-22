#!/usr/bin/env python3
"""Audit complete Potsdam embedding exports for every supported model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


DEFAULT_MODELS = (
    "olmoearth_base",
    "dinov3_vitl16",
    "copernicusfm_base",
    "presto",
    "prithviv2_300m",
    "tessera",
    "croma_base",
    "galileo_base",
    "satlas_base",
    "universat_base",
    "clay_large",
    "anysat_base",
    "panopticon_vitb14",
    "terramind_base",
)
EXPECTED_COUNTS = {"train": 3456, "val": 2016}
ERROR_PATTERN = re.compile(
    r"Traceback|OutOfMemory(?:Error)?|CUDA error|\bKilled\b|\bnan\b|\binf\b",
    re.IGNORECASE,
)


def _load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(value).__name__}.")
    return value


def _sample_indices(count: int) -> tuple[int, ...]:
    return tuple(sorted({0, count // 2, count - 1}))


def audit_split(root: Path, model: str, split: str) -> dict[str, Any]:
    print(f"Auditing {model}/{split}", flush=True)
    expected_count = EXPECTED_COUNTS[split]
    manifest_path = root / model / f"{split}.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        return {
            "split": split,
            "manifest": str(manifest_path),
            "ok": False,
            "errors": ["manifest_missing"],
        }

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        samples = []
        errors.append("samples_not_list")
    if manifest.get("format") != "geofm_embedding_manifest_v1":
        errors.append(f"unexpected_format:{manifest.get('format')!r}")
    if manifest.get("split") != split:
        errors.append(f"unexpected_split:{manifest.get('split')!r}")
    if manifest.get("count") != expected_count:
        errors.append(
            f"count:{manifest.get('count')!r}!={expected_count}"
        )
    if len(samples) != expected_count:
        errors.append(f"sample_records:{len(samples)}!={expected_count}")

    sample_ids: set[str] = set()
    embedding_files = 0
    label_files = 0
    for index, sample in enumerate(samples):
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"sample[{index}]:invalid_sample_id")
        elif sample_id in sample_ids:
            errors.append(f"sample[{index}]:duplicate_sample_id:{sample_id}")
        else:
            sample_ids.add(sample_id)

        embedding_rel = sample.get("embedding_path")
        label_rel = sample.get("label_path")
        if isinstance(embedding_rel, str) and (root / model / embedding_rel).is_file():
            embedding_files += 1
        else:
            errors.append(f"sample[{index}]:embedding_missing:{embedding_rel!r}")
        if isinstance(label_rel, str) and (root / model / label_rel).is_file():
            label_files += 1
        else:
            errors.append(f"sample[{index}]:label_missing:{label_rel!r}")

        if sample.get("source_shape") != [512, 512]:
            errors.append(
                f"sample[{index}]:source_shape:{sample.get('source_shape')!r}"
            )
        if sample.get("model_input_shape") != [64, 64]:
            errors.append(
                "sample[{}]:model_input_shape:{!r}".format(
                    index, sample.get("model_input_shape")
                )
            )
        if sample.get("label_shape") != [512, 512]:
            errors.append(
                f"sample[{index}]:label_shape:{sample.get('label_shape')!r}"
            )

    checked_tensors: list[dict[str, Any]] = []
    for index in _sample_indices(len(samples)) if samples else ():
        sample = samples[index]
        embedding_path = root / model / sample["embedding_path"]
        label_path = root / model / sample["label_path"]
        try:
            embedding = _load_tensor(embedding_path)
            label = _load_tensor(label_path)
            expected_embedding_shape = tuple(sample.get("embedding_shape", ()))
            expected_label_shape = tuple(sample.get("label_shape", ()))
            if tuple(embedding.shape) != expected_embedding_shape:
                errors.append(
                    f"sample[{index}]:embedding_tensor_shape:"
                    f"{tuple(embedding.shape)}!={expected_embedding_shape}"
                )
            if tuple(label.shape) != expected_label_shape:
                errors.append(
                    f"sample[{index}]:label_tensor_shape:"
                    f"{tuple(label.shape)}!={expected_label_shape}"
                )
            if not torch.isfinite(embedding).all().item():
                errors.append(f"sample[{index}]:embedding_non_finite")
            checked_tensors.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "embedding_shape": list(embedding.shape),
                    "label_shape": list(label.shape),
                    "embedding_dtype": str(embedding.dtype),
                    "label_dtype": str(label.dtype),
                }
            )
        except Exception as exc:  # include corrupt/truncated tensor details
            errors.append(f"sample[{index}]:tensor_load:{exc}")

    log_path = root / "logs" / f"{model}_{split}.log"
    log_errors: list[str] = []
    log_complete = False
    if log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        log_errors = [
            line.strip()
            for line in log_text.splitlines()
            if ERROR_PATTERN.search(line)
        ][:20]
        log_complete = f"Manifest: {manifest_path}" in log_text
        if log_errors:
            errors.append(f"log_errors:{len(log_errors)}")
        if not log_complete:
            errors.append("log_missing_manifest_completion")
    else:
        errors.append("log_missing")

    return {
        "split": split,
        "manifest": str(manifest_path),
        "count": manifest.get("count"),
        "sample_records": len(samples),
        "embedding_files": embedding_files,
        "label_files": label_files,
        "checked_tensors": checked_tensors,
        "log": str(log_path),
        "log_complete": log_complete,
        "log_error_lines": log_errors,
        "ok": not errors,
        "errors": errors[:100],
        "error_count": len(errors),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    report: dict[str, Any] = {
        "root": str(root),
        "expected_counts": EXPECTED_COUNTS,
        "models": {},
    }
    for model in args.models:
        splits = {
            split: audit_split(root, model, split)
            for split in EXPECTED_COUNTS
        }
        report["models"][model] = {
            "ok": all(item["ok"] for item in splits.values()),
            "splits": splits,
        }
    report["ok"] = all(item["ok"] for item in report["models"].values())
    output_path = args.output or root / "audit_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Audit report: {output_path}")
    print(f"Overall status: {'PASS' if report['ok'] else 'FAIL'}")
    for model, result in report["models"].items():
        print(f"{model}: {'PASS' if result['ok'] else 'FAIL'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
