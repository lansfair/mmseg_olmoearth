from __future__ import annotations

from mmengine.hooks import Hook
from mmseg.registry import HOOKS
from torch import nn


@HOOKS.register_module()
class GeoFMFreezeBackboneHook(Hook):
    """Enforce strict or staged linear-probe backbone trainability."""

    priority = "NORMAL"

    def __init__(
        self,
        unfreeze_epoch: int | None = None,
        strict: bool = True,
    ) -> None:
        if unfreeze_epoch is not None and unfreeze_epoch < 0:
            raise ValueError("unfreeze_epoch cannot be negative.")
        self.unfreeze_epoch = unfreeze_epoch
        self.strict = strict
        self._is_frozen = False

    @staticmethod
    def _model(runner):
        return (
            runner.model.module
            if hasattr(runner.model, "module")
            else runner.model
        )

    def _backbone(self, runner):
        return self._model(runner).backbone

    @staticmethod
    def _set_frozen(backbone, frozen: bool) -> None:
        if hasattr(backbone, "set_frozen"):
            backbone.set_frozen(frozen)
        else:
            backbone.requires_grad_(not frozen)
            backbone.eval() if frozen else backbone.train()

    def _audit(self, runner) -> None:
        model = self._model(runner)
        backbone_trainable = sum(
            parameter.numel()
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        )
        head_trainable = sum(
            parameter.numel()
            for parameter in model.decode_head.parameters()
            if parameter.requires_grad
        )
        runner.logger.info(
            "GeoFM trainability: backbone=%d, decode_head=%d parameters.",
            backbone_trainable,
            head_trainable,
        )
        if self.strict and self._is_frozen and backbone_trainable:
            raise RuntimeError(
                "Strict linear probe requires zero trainable backbone "
                "parameters."
            )
        if self.strict and head_trainable == 0:
            raise RuntimeError("Linear probe head has no trainable parameters.")
        trainable_layers = [
            (name, module)
            for name, module in model.decode_head.named_modules()
            if any(
                parameter.requires_grad
                for parameter in module.parameters(recurse=False)
            )
        ]
        affine_types = (nn.Linear, nn.Conv2d)
        if self.strict and (
            len(trainable_layers) != 1
            or not isinstance(trainable_layers[0][1], affine_types)
        ):
            names = ", ".join(name or "<root>" for name, _ in trainable_layers)
            raise RuntimeError(
                "Strict linear probe requires exactly one trainable affine "
                f"layer; found: {names or '<none>'}."
            )

    def before_train(self, runner) -> None:
        frozen = self.unfreeze_epoch != 0
        self._set_frozen(self._backbone(runner), frozen)
        self._is_frozen = frozen
        self._audit(runner)

    def before_train_epoch(self, runner) -> None:
        if (
            self._is_frozen
            and self.unfreeze_epoch is not None
            and runner.epoch >= self.unfreeze_epoch
        ):
            self._set_frozen(self._backbone(runner), False)
            self._is_frozen = False
            runner.logger.info(
                "Unfroze model.backbone at epoch %d.",
                runner.epoch,
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
