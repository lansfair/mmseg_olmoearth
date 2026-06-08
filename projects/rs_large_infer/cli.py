from __future__ import annotations

import argparse

from mmengine.config import DictAction

from .runner import run_large_inference


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shared sliding-window large GeoTIFF inference for single-input "
            "MMSeg semantic segmentation configs."
        )
    )
    parser.add_argument("image", help="Input large GeoTIFF.")
    parser.add_argument("config", help="MMSeg semantic segmentation config.")
    parser.add_argument("checkpoint", help="MMSeg segmentation checkpoint.")
    parser.add_argument("output", help="Output label GeoTIFF.")
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--band-indices",
        type=int,
        nargs="+",
        default=None,
        help="1-based input GeoTIFF bands to read in standard/Copernicus modes.",
    )
    parser.add_argument(
        "--band-scales",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-loaded-band scale factors in standard/Copernicus modes.",
    )
    parser.add_argument(
        "--rgb-channel-order",
        choices=["RGB", "BGR"],
        default=None,
        help="Band order for OLMoEarth RGB adapter GeoTIFF input.",
    )
    parser.add_argument(
        "--input-value-range",
        choices=["auto", "0_255", "0_1", "s2"],
        default=None,
        help="RGBToOlmoEarthS2 input value range override.",
    )
    parser.add_argument(
        "--source-band-names",
        nargs="+",
        default=None,
        help="Input S2 band names. Used to reorder bands to OLMoEarth order.",
    )
    parser.add_argument(
        "--s2-scale-factor",
        type=float,
        default=None,
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
        default=None,
        help="Sensing date as YYYYMMDD. Overrides date parsed from filename.",
    )
    parser.add_argument(
        "--copernicus-date-days",
        type=float,
        default=None,
        help="Sensing time as days since 1970-01-01. Overrides date parsing.",
    )
    parser.add_argument(
        "--copernicus-lon-lat",
        type=float,
        nargs=2,
        metavar=("LON", "LAT"),
        default=None,
        help="Override geospatial lon/lat metadata for every window.",
    )
    parser.add_argument(
        "--copernicus-patch-area",
        type=float,
        default=None,
        help="Copernicus patch area metadata override.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, same syntax as tools/test.py.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_large_inference(parse_args(argv))
