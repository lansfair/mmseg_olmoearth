from __future__ import annotations

from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class BackboneFreezeSwitchHook(Hook):
    """Freeze DINOv3 for the first N epochs, then unfreeze it.

    For distributed training, build DDP while the backbone is initially
    trainable and set ``find_unused_parameters=True`` in the phase-switching
    config. This keeps all reducer hooks available when parameters are later
    unfrozen.
    """

    priority = 'VERY_HIGH'

    def __init__(self, freeze_epochs: int = 10) -> None:
        self.freeze_epochs = int(freeze_epochs)
        self._last_state = None

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, 'module') else model

    def _apply(self, runner) -> None:
        model = self._unwrap(runner.model)
        backbone = getattr(model, 'backbone', None)
        if backbone is None or not hasattr(backbone, 'set_frozen'):
            raise AttributeError(
                'BackboneFreezeSwitchHook requires model.backbone.set_frozen().'
            )
        frozen = int(runner.epoch) < self.freeze_epochs
        if frozen != self._last_state:
            backbone.set_frozen(frozen)
            self._last_state = frozen
            state = 'frozen' if frozen else 'trainable'
            runner.logger.info(
                f'[BackboneFreezeSwitchHook] epoch={runner.epoch + 1}: '
                f'DINOv3 backbone is {state}.'
            )

    def before_train(self, runner) -> None:
        self._apply(runner)

    def before_train_epoch(self, runner) -> None:
        self._apply(runner)
