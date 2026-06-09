"""Core large-image inference implementation for remote-sensing MMSeg projects."""

from .cli import main, parse_args
from .runner import run_large_inference

__all__ = ["main", "parse_args", "run_large_inference"]
