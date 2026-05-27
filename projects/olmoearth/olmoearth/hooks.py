from __future__ import annotations

from mmengine.hooks import Hook
from mmseg.registry import HOOKS


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
