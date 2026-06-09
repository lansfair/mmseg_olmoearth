import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from projects.rs_large_infer.src.cli import main
from projects.rs_large_infer.tools.preset_argv import build_cli_argv


# DINOv3 + Potsdam RGB 大图推理默认参数。
# 这里的值会作为默认值；命令行传入同名参数时，会覆盖这里的默认值。
IMAGE = "/mnt/ht2-nas2/EO_test/dataset/Segmentation/Potsdam/2_Ortho_RGB/top_potsdam_5_15_RGB.tif"
CHECKPOINT = "/mnt/ht2-nas2/EO_test/openmmlab-archive/src/v1/mmseg/projects/dinov3/potsdam/checkpoints1/potsdam_dinov3-fm-base_upernet_e50-frozen.pth"
OUTPUT = "/mnt/ht2-nas2/EO_test/rs_infer_save/out/top_potsdam_5_15_RGB_dinov3_upernet.tif"

CONFIG_FILE = "/mnt/ht2-nas2/wj/large_tif_infer_test/mmseg_olmoearth/projects/dinov3/configs/potsdam/dinov3-vitl16_upernet_4xb4-50e_potsdam-rgb.py"

INPUT_MODE = "standard"
WINDOW_SIZE = (512, 512)
STRIDE = (256, 256)
BATCH_SIZE = 8
DEVICE = "cuda:0"

# 普通 Potsdam RGB GeoTIFF 默认读取前 3 个 band。
BAND_INDICES = [1, 2, 3]
BAND_SCALES = None

# 如果 config 中的 DINOv3 预训练权重路径不适用于当前服务器，可在这里覆盖。
# 示例：
# CFG_OPTIONS = {
#     "model.backbone.weights": "/checkpoints/dinov3_vitl16_pretrain_sat493m.pth",
# }
CFG_OPTIONS = None


def build_script_defaults() -> dict:
    """构造 DINOv3 Potsdam RGB 场景的大图推理默认参数。"""

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
        band_indices=BAND_INDICES,
        band_scales=BAND_SCALES,
        cfg_options=CFG_OPTIONS,
    )


if __name__ == "__main__":
    main(argv=build_cli_argv(build_script_defaults()) + sys.argv[1:])
