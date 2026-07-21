#!/usr/bin/env python
"""K-means for one dataset/model PT bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.geofm_embeddings.evaluation.bundle_tasks import run_kmeans  # noqa: E402
from projects.geofm_embeddings.evaluation.bundle import (  # noqa: E402
    evaluation_output_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run K-means on generic PT bundles.")
    parser.add_argument("dataset")
    parser.add_argument("model")
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--per-class", type=int, default=256)
    parser.add_argument("--ignore-label", type=int, default=-1)
    parser.add_argument("--clusters", type=int)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_kmeans(
        root=args.embedding_root,
        dataset=args.dataset,
        model=args.model,
        output_dir=evaluation_output_directory(
            args.output_root, dataset=args.dataset, model=args.model, task="kmeans"
        ),
        split_name=args.split,
        per_class=args.per_class,
        ignore_label=args.ignore_label,
        clusters=args.clusters,
        n_init=args.n_init,
        max_iter=args.max_iter,
    )
    print(report)


if __name__ == "__main__":
    main()
