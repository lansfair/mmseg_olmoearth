from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _get_backbone(model):
    model = _unwrap_model(model)
    if hasattr(model, 'backbone'):
        return model.backbone
    raise AttributeError('The segmentor has no .backbone attribute.')


def _count_trainable(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


@HOOKS.register_module(force=True)
class FreezeBackboneBeforeEpochHook(Hook):
    """Freeze backbone before training and unfreeze from ``unfreeze_epoch``.

    IMPORTANT: use this hook only when the model was constructed with
    ``freeze_backbone=False``. This lets the optimizer include backbone params
    from the beginning, then the hook temporarily disables gradients.
    """

    priority = 'VERY_HIGH'

    def __init__(self, unfreeze_epoch: int = 10, freeze_at_start: bool = True):
        self.unfreeze_epoch = int(unfreeze_epoch)
        self.freeze_at_start = freeze_at_start
        self._is_unfrozen = False

    def _set_trainable(self, runner, trainable: bool):
        backbone = _get_backbone(runner.model)
        if hasattr(backbone, 'set_backbone_trainable'):
            backbone.set_backbone_trainable(trainable)
        else:
            for p in backbone.parameters():
                p.requires_grad = trainable
            if not trainable:
                backbone.eval()
        total, trainable_num = _count_trainable(backbone)
        cfm = getattr(backbone, 'cfm', None)
        if cfm is not None:
            cfm_total, cfm_trainable = _count_trainable(cfm)
            runner.logger.info(
                f'[FreezeBackboneBeforeEpochHook] backbone trainable={trainable}; '
                f'backbone trainable params={trainable_num:,}/{total:,}; '
                f'cfm trainable params={cfm_trainable:,}/{cfm_total:,}'
            )
        else:
            runner.logger.info(
                f'[FreezeBackboneBeforeEpochHook] backbone trainable={trainable}; '
                f'trainable params={trainable_num:,}/{total:,}'
            )

    def before_train(self, runner):
        if self.freeze_at_start:
            self._set_trainable(runner, False)

    def before_train_epoch(self, runner):
        if (not self._is_unfrozen) and runner.epoch >= self.unfreeze_epoch:
            self._set_trainable(runner, True)
            self._is_unfrozen = True
