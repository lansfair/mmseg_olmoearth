import math
from mmengine.hooks import Hook
from mmseg.registry import HOOKS

@HOOKS.register_module()
class OlmoEarthFreezeUnfreezeHook(Hook):
    priority = 'VERY_HIGH'

    def __init__(self,
                 freeze_epochs=None,
                 freeze_epoch_fraction=0.2,
                 Unfreeze_lr_factor=0.1,
                 backbone_attr='backbone'):
        self.freeze_epochs=freeze_epochs
        self.freeze_epoch_fraction = freeze_epoch_fraction
        self.Unfreeze_lr_factor = Unfreeze_lr_factor
        self._resolved_freeze_epochs = None
        self._unfrozen = False
        self.backbone_attr = backbone_attr

    def _unwrap_model(self,model):
        return model.module if hasattr(model, 'module') else model
    def _set_backbone_trainable(self,runner,trainable):
        model = self._unwrap_model(runner.model)
        if not hasattr(model, self.backbone_attr):
            raise AttributeError(
                f'Model has no attribute "{self.backbone_attr}" to freeze.'
            )
        backbone = getattr(model, self.backbone_attr)
        for param in backbone.parameters():
            param.requires_grad = trainable
    def _resolve_freeze_epochs(self, runner):
        if self.freeze_epochs is not None:
            return int(self.freeze_epochs)
        max_epochs = getattr(runner.train_loop, 'max_epochs', None)
        if max_epochs is None:
            raise ValueError(
                'freeze_epoch_fraction requires EpochBasedTrainLoop.'
                'Set freeze_epochs explicitly for IterBasedTrainLoop.'
            )
        return int(math.ceil(max_epochs * self.freeze_epoch_fraction))
    
    def _scale_global_lr(self, runner):
        optimizer = runner.optim_wrapper.optimizer
        for group in optimizer.param_groups:
            group['lr'] *= self.Unfreeze_lr_factor
        return [group['lr'] for group in optimizer.param_groups]

    def before_train(self, runner):
            self._resolved_freeze_epochs = self._resolve_freeze_epochs(runner)

            if runner.epoch >= self._resolved_freeze_epochs:
                self._unfrozen = True
                self._set_backbone_trainable(runner, trainable=True)
                runner.logger.info(
                    f'{self.backbone_attr} is trainable because runner.epoch'
                    f'({runner.epoch}) >= freeze_epochs'
                    f'({self._resolved_freeze_epochs}).'
                )
            else:
                self._unfrozen = False
                self._set_backbone_trainable(runner, trainable=False)
                runner.logger.info(
                    f'Freeze {self.backbone_attr} for first '
                    f'{self._resolved_freeze_epochs} epochs'
                    
                )
    def before_train_epoch(self, runner):
        if self._unfrozen:
            return
        if runner.epoch >= self._resolved_freeze_epochs:
            self._set_backbone_trainable(runner, trainable=True)
            self._unfrozen = True
            lrs = self._scale_global_lr(runner)
            runner.logger.info(
                f'Unfreeze {self.backbone_attr} at epoch {runner.epoch}'
                f'multiply global LR by {self.Unfreeze_lr_factor}. '
                f'Current LRs: {lrs}. '
            )

# FreezeBackEpochHook = OlmoEarthFreezeUnfreezeHook           