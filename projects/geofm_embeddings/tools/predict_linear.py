#!/usr/bin/env python
"""Run a trained dense linear probe and save index-label PNG predictions."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.geofm_embeddings.evaluation.bundle import write_json_atomic  # noqa: E402
from projects.geofm_embeddings.evaluation.linear import DenseLinearProbe  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict dense segmentation labels from an embedding PT bundle."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("input", type=Path, help="Labeled or unlabeled embedding PT")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help="Override source_shape metadata for every output PNG.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_embeddings(path: Path) -> tuple[torch.Tensor, list[str], torch.Tensor | None]:
    payload = torch.load(
        path.resolve(), map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(payload, dict) or not isinstance(
        payload.get("embeddings"), torch.Tensor
    ):
        raise TypeError("Input PT must contain an embeddings tensor")
    embeddings = payload["embeddings"]
    if embeddings.ndim != 4:
        raise ValueError(
            f"Dense prediction requires [N,H,W,D] embeddings, got {embeddings.shape}"
        )
    layout = str(payload.get("embedding_layout", "NHWD"))
    if layout == "NDHW":
        embeddings = embeddings.permute(0, 2, 3, 1)
    elif layout != "NHWD":
        raise ValueError(f"Unsupported embedding_layout {layout!r}")
    if not embeddings.dtype.is_floating_point:
        raise TypeError("Embeddings must be floating point")

    raw_ids = payload.get("sample_ids")
    if raw_ids is None:
        sample_ids = [f"{index:08d}" for index in range(len(embeddings))]
    else:
        sample_ids = [str(value) for value in raw_ids]
        if len(sample_ids) != len(embeddings):
            raise ValueError("sample_ids length differs from embeddings")

    source_shapes = payload.get("source_shapes")
    if source_shapes is not None:
        if not isinstance(source_shapes, torch.Tensor) or source_shapes.shape != (
            len(embeddings),
            2,
        ):
            raise ValueError("source_shapes must be an [N,2] tensor")
        source_shapes = source_shapes.to(torch.int64)
        if bool((source_shapes < 1).any()):
            raise ValueError("source_shapes must be positive")
    return embeddings, sample_ids, source_shapes


def _load_probe(path: Path, device: torch.device) -> tuple[DenseLinearProbe, torch.Tensor]:
    payload = torch.load(path.resolve(), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "geofm_linear_probe_v1":
        raise ValueError("Unsupported linear-probe checkpoint")
    if payload.get("mode") != "dense_segmentation":
        raise ValueError("predict_linear.py requires a dense segmentation probe")
    class_values = payload.get("class_values")
    state_dict = payload.get("state_dict")
    if not isinstance(class_values, torch.Tensor) or not isinstance(state_dict, dict):
        raise TypeError("Checkpoint is missing class_values or state_dict")
    class_values = class_values.to(torch.int64)
    probe = DenseLinearProbe(int(payload["in_features"]), len(class_values))
    probe.load_state_dict(state_dict, strict=True)
    return probe.to(device).eval(), class_values.to(device)


def _safe_stem(sample_id: str, index: int, used: set[str]) -> str:
    stem = re.sub(r"[^0-9A-Za-z_.-]+", "_", sample_id).strip("._")
    if not stem:
        stem = f"{index:08d}"
    candidate = stem
    suffix = 1
    while candidate in used:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _save_png(path: Path, prediction: torch.Tensor) -> None:
    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError("PNG output requires Pillow") from error
    array = prediction.detach().cpu().numpy()
    minimum, maximum = int(array.min()), int(array.max())
    if minimum >= 0 and maximum <= 255:
        array = array.astype(np.uint8, copy=False)
        mode = "L"
    elif minimum >= 0 and maximum <= 65535:
        array = array.astype(np.uint16, copy=False)
        mode = "I;16"
    else:
        array = array.astype(np.int32, copy=False)
        mode = "I"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".png.tmp")
    Image.fromarray(array, mode=mode).save(temporary, format="PNG")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    embeddings, sample_ids, source_shapes = _load_embeddings(args.input)
    probe, class_values = _load_probe(args.checkpoint, device)
    if embeddings.shape[-1] != probe.linear.in_features:
        raise ValueError(
            f"Embedding D={embeddings.shape[-1]} differs from probe "
            f"D={probe.linear.in_features}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    used_names: set[str] = set()
    with torch.inference_mode():
        for start in range(0, len(embeddings), args.batch_size):
            batch = embeddings[start : start + args.batch_size].to(device).float()
            logits = probe.linear(batch).permute(0, 3, 1, 2)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(f"Non-finite logits in batch starting {start}")
            for offset, item_logits in enumerate(logits):
                index = start + offset
                if args.output_size is not None:
                    output_size = tuple(args.output_size)
                elif source_shapes is not None:
                    output_size = tuple(int(value) for value in source_shapes[index])
                else:
                    output_size = tuple(map(int, item_logits.shape[-2:]))
                if tuple(item_logits.shape[-2:]) != output_size:
                    item_logits = F.interpolate(
                        item_logits[None],
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                prediction = class_values[item_logits.argmax(dim=0)]
                stem = _safe_stem(sample_ids[index], index, used_names)
                path = output_dir / f"{stem}.png"
                if path.exists() and not args.overwrite:
                    raise FileExistsError(f"Refusing to overwrite {path}")
                _save_png(path, prediction)
                records.append(
                    {
                        "sample_id": sample_ids[index],
                        "prediction_path": path.name,
                        "shape": list(output_size),
                    }
                )
            print(f"\rPredicted {len(records)}/{len(embeddings)}", end="", flush=True)
    print()
    manifest = {
        "format": "geofm_linear_predictions_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "input": str(args.input.resolve()),
        "count": len(records),
        "class_values": class_values.cpu().tolist(),
        "samples": records,
    }
    report = write_json_atomic(output_dir / "predictions.json", manifest)
    print(report)


if __name__ == "__main__":
    main()
