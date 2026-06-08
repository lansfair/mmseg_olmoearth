from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from projects.rs_large_infer.cli import main, parse_args
    from projects.rs_large_infer.runner import run_large_inference
else:
    from .cli import main, parse_args
    from .runner import run_large_inference

__all__ = ["main", "parse_args", "run_large_inference"]


if __name__ == "__main__":
    main()
