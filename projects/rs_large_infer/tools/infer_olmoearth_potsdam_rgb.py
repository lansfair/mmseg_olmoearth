import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from projects.rs_large_infer.src.cli import main
from projects.rs_large_infer.tools.preset_argv import build_cli_argv


# OLMoEarth + Potsdam RGB 大图推理默认参数。
# 这里的值会作为默认值；命令行传入同名参数时，会覆盖这里的默认值。
IMAGE = "/mnt/ht2-nas2/EO_test/dataset/Segmentation/Potsdam/2_Ortho_RGB/top_potsdam_5_15_RGB.tif"
CHECKPOINT = "/mnt/ht2-nas2/EO_test/openmmlab-archive/src/v1/mmseg/projects/olmoearth/potsdam/checkpoints/olmoearth-base_upernet_4xb4-80k_potsdam-rgb-p16-512x512_unfreeze.pth"
OUTPUT = "/mnt/ht2-nas2/EO_test/rs_infer_save/out/top_potsdam_5_15_RGB.tif"

CONFIG_FILE = "/mnt/ht2-nas2/EO_test/openmmlab-archive/src/v1/mmseg/projects/olmoearth/potsdam/configs/olmoearth-base_upernet_4xb4-80k_potsdam-rgb-p16-512x512_unfreeze.py"

INPUT_MODE = "rgb"
WINDOW_SIZE = (512, 512)
STRIDE = (256, 256)
BATCH_SIZE = 8
DEVICE = "cuda:0"

# 大多数 GeoTIFF 是 RGB 顺序；如果你的输入实际是 BGR，可用命令行覆盖：
# --rgb-channel-order BGR
RGB_CHANNEL_ORDER = "RGB"
INPUT_VALUE_RANGE = None
TIMESTAMP = (1, 1, 2025)

# 如果 config 中的 OLMoEarth backbone 路径不适用于当前服务器，可在这里覆盖。
# 示例：
# CFG_OPTIONS = {
#     "model.backbone.model_config_path": "/checkpoints/OlmoEarth-v1-Base/config.json",
#     "model.backbone.init_cfg.checkpoint": "/checkpoints/OlmoEarth-v1-Base/pytorch_model.bin",
# }
CFG_OPTIONS = None


def build_script_defaults() -> dict:
    """构造 OLMoEarth Potsdam RGB 场景的大图推理默认参数。"""

    return dict(
        image=IMAGE,
        config=CONFIG_FILE,
        checkpoint=CHECKPOINT,
        output=OUTPUT,
        input_mode=INPUT_MODE,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        rgb_channel_order=RGB_CHANNEL_ORDER,
        input_value_range=INPUT_VALUE_RANGE,
        timestamp=TIMESTAMP,
        cfg_options=CFG_OPTIONS,
    )


if __name__ == "__main__":
    main(argv=build_cli_argv(build_script_defaults()) + sys.argv[1:])
