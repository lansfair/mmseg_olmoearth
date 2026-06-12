from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class OlmoEarthFTStrategyHook(Hook):
    def __init__(self, total_epoch, unfrozen_epoch_rate=0.8) -> None:
        if not total_epoch > 0:
            raise
        if unfrozen_epoch_rate < 0 or unfrozen_epoch_rate > 1:
            raise
        self.frozen_epoch = int(total_epoch*(1-unfrozen_epoch_rate))

    def before_train_epoch(self, runner) -> None:
        if runner.model.backbone.is_frozen:
            if runner.epoch > self.frozen_epoch:
                runner.model.backbone.unfreeze()
        elif runner.epoch < self.frozen_epoch:
            runner.model.backbone.freeze()
        return super().before_train_epoch(runner)
