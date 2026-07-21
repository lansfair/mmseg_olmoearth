#!/usr/bin/env python
"""Cosine semantic retrieval for one dataset/model PT bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.geofm_embeddings.evaluation.bundle_tasks import (  # noqa: E402
    run_cosine_retrieval,
)
from projects.geofm_embeddings.evaluation.bundle import (  # noqa: E402
    evaluation_output_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cosine semantic retrieval on generic PT bundles."
    )
    parser.add_argument("dataset")
    parser.add_argument("model")
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--gallery-split", choices=("train", "valid", "test"), default="train"
    )
    parser.add_argument(
        "--query-split", choices=("train", "valid", "test"), default="test"
    )
    parser.add_argument("--gallery-per-class", type=int, default=512)
    parser.add_argument("--query-per-class", type=int, default=256)
    parser.add_argument("--ignore-label", type=int, default=-1)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_cosine_retrieval(
        root=args.embedding_root,
        dataset=args.dataset,
        model=args.model,
        output_dir=evaluation_output_directory(
            args.output_root,
            dataset=args.dataset,
            model=args.model,
            task="cosine_retrieval",
        ),
        gallery_split=args.gallery_split,
        query_split=args.query_split,
        gallery_per_class=args.gallery_per_class,
        query_per_class=args.query_per_class,
        ignore_label=args.ignore_label,
        k_values=args.k_values,
        batch_size=args.batch_size,
    )
    print(report)


if __name__ == "__main__":
    main()
