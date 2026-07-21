#!/usr/bin/env python
"""OLMoEarth-style cosine kNN for one dataset/model PT bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.geofm_embeddings.evaluation.bundle import (  # noqa: E402
    evaluation_output_directory,
)
from projects.geofm_embeddings.evaluation.knn import run_knn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OLMoEarth-style cosine kNN on classification bundles."
    )
    parser.add_argument("dataset")
    parser.add_argument("model")
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("valid", "test"),
        default="test",
        help="Query split; train is always used as the kNN gallery",
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--ignore-label", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_knn(
        root=args.embedding_root,
        dataset=args.dataset,
        model=args.model,
        output_dir=evaluation_output_directory(
            args.output_root, dataset=args.dataset, model=args.model, task="knn"
        ),
        split_name=args.split,
        k=args.k,
        temperature=args.temperature,
        batch_size=args.batch_size,
        ignore_label=args.ignore_label,
        device=args.device,
    )
    print(report)


if __name__ == "__main__":
    main()
