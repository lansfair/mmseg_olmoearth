from __future__ import annotations


def build_cli_argv(defaults: dict) -> list[str]:
    """把 preset 默认参数转换成标准命令行参数列表。"""

    argv = [
        "--image",
        str(defaults["image"]),
        "--config",
        str(defaults["config"]),
        "--checkpoint",
        str(defaults["checkpoint"]),
        "--output",
        str(defaults["output"]),
        "--input-mode",
        str(defaults["input_mode"]),
        "--batch-size",
        str(defaults["batch_size"]),
        "--device",
        str(defaults["device"]),
    ]

    if defaults.get("window_size") is not None:
        argv += ["--window-size", *[str(value) for value in defaults["window_size"]]]
    if defaults.get("stride") is not None:
        argv += ["--stride", *[str(value) for value in defaults["stride"]]]
    if defaults.get("band_indices") is not None:
        argv += ["--band-indices", *[str(value) for value in defaults["band_indices"]]]
    if defaults.get("band_scales") is not None:
        argv += ["--band-scales", *[str(value) for value in defaults["band_scales"]]]
    if defaults.get("rgb_channel_order") is not None:
        argv += ["--rgb-channel-order", str(defaults["rgb_channel_order"])]
    if defaults.get("input_value_range") is not None:
        argv += ["--input-value-range", str(defaults["input_value_range"])]
    if defaults.get("source_band_names") is not None:
        argv += ["--source-band-names", *[str(value) for value in defaults["source_band_names"]]]
    if defaults.get("s2_scale_factor") is not None:
        argv += ["--s2-scale-factor", str(defaults["s2_scale_factor"])]
    if defaults.get("timestamp") is not None:
        argv += ["--timestamp", *[str(value) for value in defaults["timestamp"]]]
    if defaults.get("copernicus_date") is not None:
        argv += ["--copernicus-date", str(defaults["copernicus_date"])]
    if defaults.get("copernicus_date_days") is not None:
        argv += ["--copernicus-date-days", str(defaults["copernicus_date_days"])]
    if defaults.get("copernicus_lon_lat") is not None:
        argv += ["--copernicus-lon-lat", *[str(value) for value in defaults["copernicus_lon_lat"]]]
    if defaults.get("copernicus_patch_area") is not None:
        argv += ["--copernicus-patch-area", str(defaults["copernicus_patch_area"])]
    if defaults.get("cfg_options") is not None:
        argv += [
            "--cfg-options",
            *[f"{key}={value}" for key, value in defaults["cfg_options"].items()],
        ]
    return argv
