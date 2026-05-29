from __future__ import annotations

import os.path as osp
from typing import Sequence

import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer
from mmseg.engine.hooks import SegVisualizationHook
from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample


@HOOKS.register_module()
class FreezeBackboneUntilEpochHook(Hook):
    """Freeze the OLMoEarth backbone, then unfreeze at a configured epoch."""

    priority = "NORMAL"

    def __init__(self, unfreeze_epoch: int | None = 0) -> None:
        self.unfreeze_epoch = unfreeze_epoch
        self._is_frozen = False

    @staticmethod
    def _set_trainable(module, trainable: bool) -> None:
        for param in module.parameters():
            param.requires_grad = trainable

    @staticmethod
    def _backbone(runner):
        model = (
            runner.model.module
            if hasattr(runner.model, "module")
            else runner.model
        )
        return model.backbone

    def before_train(self, runner) -> None:
        if self.unfreeze_epoch == 0:
            return
        backbone = self._backbone(runner)
        self._set_trainable(backbone, False)
        backbone.eval()
        self._is_frozen = True
        runner.logger.info("Frozen model.backbone before OLMoEarth training.")

    def before_train_epoch(self, runner) -> None:
        if self.unfreeze_epoch is None:
            if self._is_frozen:
                self._backbone(runner).eval()
            return
        if self._is_frozen and runner.epoch >= self.unfreeze_epoch:
            backbone = self._backbone(runner)
            self._set_trainable(backbone, True)
            backbone.train()
            self._is_frozen = False
            runner.logger.info(
                f"Unfroze model.backbone at epoch {runner.epoch}."
            )
        elif self._is_frozen:
            self._backbone(runner).eval()

    def before_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch=None,
    ) -> None:
        if self._is_frozen:
            self._backbone(runner).eval()


def _normalize_band_name(value: str) -> str:
    return str(value).strip().upper().replace("_", "").replace(" ", "")


def _as_first_tensor(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs[0]
    if isinstance(inputs, (list, tuple)):
        return inputs[0]
    raise TypeError(f"Unsupported visualization inputs: {type(inputs)}")


def _stretch_to_uint8(image: np.ndarray) -> np.ndarray:
    out = np.empty(image.shape, dtype=np.uint8)
    for channel_idx in range(image.shape[-1]):
        channel = image[..., channel_idx].astype(np.float32, copy=False)
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            out[..., channel_idx] = 0
            continue
        lo, hi = np.percentile(finite, [2, 98])
        if hi <= lo:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
        if hi <= lo:
            out[..., channel_idx] = 0
            continue
        scaled = (channel - lo) / (hi - lo)
        out[..., channel_idx] = np.clip(scaled * 255.0, 0, 255).astype(
            np.uint8
        )
    return out


@HOOKS.register_module()
class OlmoEarthVisualizationHook(SegVisualizationHook):
    """Visualize OLMoEarth array inputs without reading multiband files."""

    def __init__(
        self,
        draw: bool = False,
        interval: int = 50,
        show: bool = False,
        wait_time: float = 0.0,
        timestep: int | str = "middle",
    ) -> None:
        super().__init__(
            draw=draw,
            interval=interval,
            show=show,
            wait_time=wait_time,
            backend_args=None,
        )
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.timestep = timestep

    def _timestep_index(self, num_timesteps: int) -> int:
        if self.timestep == "middle":
            return max(num_timesteps // 2, 0)
        index = int(self.timestep)
        return max(0, min(index, num_timesteps - 1))

    def _channel_indices(
        self,
        data_sample: SegDataSample,
        num_channels: int,
    ) -> list[int]:
        meta = data_sample.metainfo
        band_names = [
            _normalize_band_name(name)
            for name in meta.get("olmoearth_band_names", [])
        ]
        num_timesteps = int(meta.get("olmoearth_num_timesteps", 1))
        t_idx = self._timestep_index(num_timesteps)
        wanted = ("B04", "B03", "B02")
        if all(band in band_names for band in wanted):
            return [
                band_names.index(band) * num_timesteps + t_idx
                for band in wanted
            ]
        wanted = ("VV", "VH")
        if all(band in band_names for band in wanted):
            vv = band_names.index("VV") * num_timesteps + t_idx
            vh = band_names.index("VH") * num_timesteps + t_idx
            return [vv, vh, vv]
        return [0, min(1, num_channels - 1), min(2, num_channels - 1)]

    def _make_image(self, inputs, data_sample: SegDataSample) -> np.ndarray:
        tensor = _as_first_tensor(inputs).detach().cpu()
        image = tensor.numpy().transpose(1, 2, 0)
        indices = self._channel_indices(data_sample, image.shape[-1])
        indices = [min(index, image.shape[-1] - 1) for index in indices]
        return _stretch_to_uint8(image[..., indices])

    @staticmethod
    def _sample_name(data_sample: SegDataSample, prefix: str) -> str:
        if "sample_id" in data_sample.metainfo:
            return f"{prefix}_{data_sample.metainfo['sample_id']}"
        if "img_paths" in data_sample.metainfo:
            return f"{prefix}_{osp.basename(data_sample.metainfo['img_paths'][0])}"
        if "img_path" in data_sample.metainfo:
            return f"{prefix}_{osp.basename(data_sample.metainfo['img_path'])}"
        return prefix

    def after_val_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict,
        outputs: Sequence[SegDataSample],
    ) -> None:
        if self.draw is False:
            return
        total_curr_iter = runner.iter + batch_idx
        if total_curr_iter % self.interval != 0:
            return
        img = self._make_image(data_batch["inputs"], outputs[0])
        window_name = self._sample_name(outputs[0], "val")
        self._visualizer.add_datasample(
            window_name,
            img,
            data_sample=outputs[0],
            show=self.show,
            wait_time=self.wait_time,
            step=total_curr_iter,
        )

    def after_test_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict,
        outputs: Sequence[SegDataSample],
    ) -> None:
        if self.draw is False:
            return
        inputs = data_batch["inputs"]
        for output_idx, data_sample in enumerate(outputs):
            self._test_index += 1
            if isinstance(inputs, torch.Tensor):
                current_inputs = inputs[output_idx: output_idx + 1]
            else:
                current_inputs = [inputs[output_idx]]
            img = self._make_image(current_inputs, data_sample)
            window_name = self._sample_name(data_sample, "test")
            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                step=self._test_index,
            )
