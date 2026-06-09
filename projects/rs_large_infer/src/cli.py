from __future__ import annotations

import argparse
import sys

from mmengine.config import DictAction
if __package__ in {None, ""}:  # pragma: no cover
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from projects.rs_large_infer.src.runner import run_large_inference
else:
    from .runner import run_large_inference


def normalize_legacy_positional_args(argv: list[str]) -> list[str]:
    """把旧式前四个位置参数转换成命名参数，减少 CLI 分支复杂度。"""

    if len(argv) < 4:
        return argv
    if any(arg.startswith("-") for arg in argv[:4]):
        return argv
    return [
        "--image",
        argv[0],
        "--config",
        argv[1],
        "--checkpoint",
        argv[2],
        "--output",
        argv[3],
        *argv[4:],
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析大图推理命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Shared sliding-window large GeoTIFF inference for single-input "
            "MMSeg semantic segmentation configs."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Input large GeoTIFF.",
    )
    parser.add_argument("--config", required=True, help="MMSeg config.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="MMSeg segmentation checkpoint.",
    )
    parser.add_argument("--output", required=True, help="Output label GeoTIFF.")
    parser.add_argument(
        "--input-mode",
        choices=["auto", "standard", "rgb", "s2", "copernicus"],
        default="auto",
        help=(
            "Input interpretation. auto selects Copernicus, OLMoEarth RGB, "
            "OLMoEarth S2, or standard mode from the config."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Sliding window size. Defaults to config crop_size/data_preprocessor size.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        help="Sliding stride. Defaults to the window size.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=2,
        help="Number of batches to prepare ahead in a background thread. Use 0 to disable.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--band-indices",
        type=int,
        nargs="+",
        help="1-based input GeoTIFF bands to read in standard/Copernicus modes.",
    )
    parser.add_argument(
        "--band-scales",
        type=float,
        nargs="+",
        help="Optional per-loaded-band scale factors in standard/Copernicus modes.",
    )
    parser.add_argument(
        "--rgb-channel-order",
        choices=["RGB", "BGR"],
        help="Band order for OLMoEarth RGB adapter GeoTIFF input.",
    )
    parser.add_argument(
        "--input-value-range",
        choices=["auto", "0_255", "0_1", "s2"],
        help="RGBToOlmoEarthS2 input value range override.",
    )
    parser.add_argument(
        "--source-band-names",
        nargs="+",
        help="Input S2 band names. Used to reorder bands to OLMoEarth order.",
    )
    parser.add_argument(
        "--s2-scale-factor",
        type=float,
        help="Optional scale factor applied to S2 windows before normalization.",
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        nargs=3,
        metavar=("DAY", "MONTH", "YEAR"),
        default=(1, 1, 2025),
        help="Timestamp metadata passed to OLMoEarth for each timestep.",
    )
    parser.add_argument(
        "--copernicus-date",
        help="Sensing date as YYYYMMDD. Overrides date parsed from filename.",
    )
    parser.add_argument(
        "--copernicus-date-days",
        type=float,
        help="Sensing time as days since 1970-01-01. Overrides date parsing.",
    )
    parser.add_argument(
        "--copernicus-lon-lat",
        type=float,
        nargs=2,
        metavar=("LON", "LAT"),
        help="Override geospatial lon/lat metadata for every window.",
    )
    parser.add_argument(
        "--copernicus-patch-area",
        type=float,
        help="Copernicus patch area metadata override.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, same syntax as tools/test.py.",
    )
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    return parser.parse_args(normalize_legacy_positional_args(raw_argv))


def main(argv: list[str] | None = None) -> None:
    """命令行入口：解析最终参数后启动大图滑窗推理。"""

    run_large_inference(parse_args(argv))
