#!/usr/bin/env python
"""DBSCAN for one dataset/model PT bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.geofm_embeddings.evaluation.bundle_tasks import run_dbscan  # noqa: E402
from projects.geofm_embeddings.evaluation.bundle import (  # noqa: E402
    evaluation_output_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DBSCAN on generic PT bundles.")
    parser.add_argument("dataset")
    parser.add_argument("model")
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--per-class", type=int, default=256)
    parser.add_argument("--ignore-label", type=int, default=-1)
    parser.add_argument("--min-samples", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument(
        "--eps-multipliers", nargs="+", type=float, default=[0.9, 1.0, 1.1]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_dbscan(
        root=args.embedding_root,
        dataset=args.dataset,
        model=args.model,
        output_dir=evaluation_output_directory(
            args.output_root, dataset=args.dataset, model=args.model, task="dbscan"
        ),
        split_name=args.split,
        per_class=args.per_class,
        ignore_label=args.ignore_label,
        min_samples=args.min_samples,
        eps_multipliers=args.eps_multipliers,
    )
    print(report)


if __name__ == "__main__":
    main()
