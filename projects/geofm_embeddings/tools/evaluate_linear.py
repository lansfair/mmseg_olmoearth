#!/usr/bin/env python
"""Linear classification or segmentation for one dataset/model PT bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.geofm_embeddings.evaluation.linear import (  # noqa: E402
    run_linear,
    run_linear_train_only,
)
from projects.geofm_embeddings.evaluation.bundle import (  # noqa: E402
    evaluation_output_directory,
)


DEFAULT_LEARNING_RATES = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a classification or segmentation linear probe."
    )
    parser.add_argument("dataset", help="Dataset directory name")
    parser.add_argument("model", help="Model directory name")
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("valid", "test"),
        default="test",
        help="Final reported split; valid is also used for model selection",
    )
    parser.add_argument(
        "--learning-rates", nargs="+", type=float, default=DEFAULT_LEARNING_RATES
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Use only train.pt; do not load valid.pt or test.pt.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Required fixed maximum learning rate for --train-only.",
    )
    parser.add_argument("--ignore-label", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_only:
        if args.learning_rate is None:
            raise ValueError("--train-only requires --learning-rate")
        report = run_linear_train_only(
            root=args.embedding_root,
            dataset=args.dataset,
            model=args.model,
            output_dir=evaluation_output_directory(
                args.output_root,
                dataset=args.dataset,
                model=args.model,
                task="linear_train_only",
            ),
            learning_rate=args.learning_rate,
            ignore_label=args.ignore_label,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_limit=args.sample_limit,
        )
        print(report)
        return
    report = run_linear(
        root=args.embedding_root,
        dataset=args.dataset,
        model=args.model,
        output_dir=evaluation_output_directory(
            args.output_root, dataset=args.dataset, model=args.model, task="linear"
        ),
        evaluation_split=args.split,
        learning_rates=args.learning_rates,
        ignore_label=args.ignore_label,
        device=args.device,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_limit=args.sample_limit,
    )
    print(report)


if __name__ == "__main__":
    main()
