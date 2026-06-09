from __future__ import annotations

from typing import Any

import numpy as np
from mmengine.config import Config
from mmengine.dataset import Compose

from ..utils import clean_pipeline, pack_window, read_window


class BaseAdapter:
    """标准适配器：处理无需额外遥感 metadata 的普通多波段输入。"""

    name = "standard"
    meta_keys = (
        "img_path",
        "ori_shape",
        "img_shape",
        "pad_shape",
        "scale_factor",
        "flip",
        "flip_direction",
        "reduce_zero_label",
    )

    def __init__(
        self,
        cfg: Config,
        raw_pipeline: list[dict[str, Any]],
        args,
    ) -> None:
        """保存配置、参数和清理后的测试 pipeline。"""

        self.cfg = cfg
        self.raw_pipeline = raw_pipeline
        self.args = args
        self.pipeline = Compose(clean_pipeline(raw_pipeline))
        self.band_indices = args.band_indices
        self.band_scales = args.band_scales
        self.nan_to_num = False
        self.to_float32 = False

    @classmethod
    def detect(cls, cfg: Config, raw_pipeline: list[dict[str, Any]]) -> bool:
        """标准适配器始终可用，作为其他模式未命中时的兜底。"""

        return True

    def prepare(self, src) -> None:
        """根据输入模式在读图前补充默认 band 设置。"""

        if self.args.input_mode == "rgb" and self.band_indices is None:
            self.band_indices = [1, 2, 3]

    def read(
        self,
        src,
        grid: tuple[int, int, int, int, int, int, int, int],
    ) -> np.ndarray:
        """从 GeoTIFF 中读取一个滑窗，并应用通用 band/缩放设置。"""

        return read_window(
            src,
            grid,
            self.band_indices,
            self.band_scales,
            self.nan_to_num,
            self.to_float32,
        )

    def make_results(
        self,
        image: np.ndarray,
        src,
        grid: tuple[int, int, int, int, int, int, int, int],
    ) -> dict[str, Any]:
        """构造送入 MMSeg transform pipeline 的基础 results 字典。"""

        return {
            "img": image,
            "img_path": self.args.image,
            "img_shape": image.shape[:2],
            "ori_shape": image.shape[:2],
        }

    def pack(
        self,
        image: np.ndarray,
        src,
        grid: tuple[int, int, int, int, int, int, int, int],
    ) -> dict[str, Any]:
        """将滑窗影像转换成 model.test_step 接收的数据项。"""

        return pack_window(
            self.make_results(image, src, grid),
            self.pipeline,
            self.meta_keys,
        )
