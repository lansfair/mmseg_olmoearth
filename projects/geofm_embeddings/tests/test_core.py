from __future__ import annotations

import pytest
import torch
from torch import nn

from projects.geofm_embeddings.geofm_embeddings import (
    BaseGeoFMAdapter,
    EmbeddingResult,
    GeoFMBackbone,
    GeoFMFreezeBackboneHook,
    GeoFMLinearHead,
    GeoFMPatchLinearHead,
    ModelCapabilities,
    PrecomputedEmbeddingBackbone,
)
from projects.geofm_embeddings.geofm_embeddings.data_preprocessor import (
    _stack_nested,
)


class DummyAdapter(BaseGeoFMAdapter):
    model_family = "dummy"

    def __init__(self) -> None:
        super().__init__(model_variant="test")
        self.out_channels = 3
        self.seen_metadata = None
        self.freeze = False
        self.proj = nn.Linear(3, 3)
        self.inner = nn.Linear(3, 3)
        self.inner.frozen = False
        self.inner.frozen_exclude = ["projection"]

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supported_modalities=frozenset({"sentinel2_l2a"}),
            required_modalities=frozenset({"sentinel2_l2a"}),
            native_stride=1,
        )

    def prepare_inputs(self, inputs, batch_metainfo=None):
        self.seen_metadata = batch_metainfo
        return self.modality_tensors(inputs)["sentinel2_l2a"]

    def extract_global(self, prepared_inputs):
        return prepared_inputs.mean(dim=(1, 3, 4))

    def extract_dense(self, prepared_inputs):
        return prepared_inputs.mean(dim=1)


def _inputs():
    return {
        "modalities": {
            "sentinel2_l2a": torch.ones(2, 4, 3, 8, 8),
        },
        "timestamps": torch.zeros(2, 4, 3, dtype=torch.long),
    }


def test_capabilities_reject_missing_modality():
    capabilities = DummyAdapter().capabilities
    with pytest.raises(ValueError, match="Missing required modalities"):
        capabilities.validate(set(), "global")


def test_embedding_result_checks_layout():
    result = EmbeddingResult(
        tensor=torch.zeros(2, 3),
        mode="global",
        model_family="dummy",
        model_variant="test",
        modalities=("sentinel2_l2a",),
    )
    assert result.embedding_dim == 3
    assert result.manifest_metadata()["l2_normalized"] is False
    with pytest.raises(ValueError, match="must have 4 dimensions"):
        EmbeddingResult(
            tensor=torch.zeros(2, 3),
            mode="dense",
            model_family="dummy",
            model_variant="test",
            modalities=("sentinel2_l2a",),
        )


def test_backbone_global_and_dense_output():
    dense = GeoFMBackbone(adapter=DummyAdapter(), output_mode="dense")
    dense.set_batch_metainfo([{"sample_id": "a"}, {"sample_id": "b"}])
    dense_output = dense(_inputs())[0]
    assert dense_output.shape == (2, 3, 8, 8)
    assert dense.adapter.seen_metadata[0]["sample_id"] == "a"

    global_backbone = GeoFMBackbone(
        adapter=DummyAdapter(),
        output_mode="global",
    )
    global_output = global_backbone(_inputs())[0]
    assert global_output.shape == (2, 3)


def test_backbone_enforces_one_freeze_policy():
    backbone = GeoFMBackbone(
        adapter=DummyAdapter(),
        output_mode="dense",
        frozen=True,
    )
    assert not any(parameter.requires_grad for parameter in backbone.parameters())
    assert backbone.adapter.freeze
    assert backbone.adapter.inner.frozen
    assert backbone.adapter.inner.frozen_exclude == []
    backbone.train()
    assert backbone.training
    assert not backbone.adapter.training

    backbone.set_frozen(False)
    assert all(parameter.requires_grad for parameter in backbone.parameters())
    assert not backbone.adapter.freeze
    assert not backbone.adapter.inner.frozen
    assert backbone.adapter.inner.frozen_exclude == ["all"]
    backbone.train()
    assert backbone.adapter.training


def test_linear_heads_have_expected_output_layout():
    common = dict(
        in_channels=3,
        channels=3,
        in_index=0,
        num_classes=2,
        align_corners=False,
    )
    features = (torch.randn(2, 3, 4, 5),)

    patch_head = GeoFMPatchLinearHead(patch_size=2, **common)
    assert patch_head(features).shape == (2, 2, 8, 10)
    assert patch_head.dropout is None
    assert sum(p.numel() for p in patch_head.parameters()) == 3 * 8 + 8

    pixel_head = GeoFMLinearHead(scale_factor=2, **common)
    assert pixel_head(features).shape == (2, 2, 8, 10)
    assert pixel_head.dropout is None
    assert sum(p.numel() for p in pixel_head.parameters()) == 3 * 2 + 2


def test_freeze_hook_enforces_strict_and_staged_probe():
    class ProbeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = GeoFMBackbone(
                adapter=DummyAdapter(),
                output_mode="dense",
            )
            self.decode_head = GeoFMLinearHead(
                in_channels=3,
                channels=3,
                in_index=0,
                num_classes=2,
            )

    class Logger:
        def info(self, *args, **kwargs):
            return None

    class Runner:
        model = ProbeModel()
        logger = Logger()
        epoch = 0

    runner = Runner()
    hook = GeoFMFreezeBackboneHook(unfreeze_epoch=2, strict=True)
    hook.before_train(runner)
    assert not any(
        parameter.requires_grad
        for parameter in runner.model.backbone.parameters()
    )

    runner.epoch = 2
    hook.before_train_epoch(runner)
    assert all(
        parameter.requires_grad
        for parameter in runner.model.backbone.parameters()
    )


def test_nested_preprocessor_stack():
    value = {
        "modalities": {
            "sentinel1": [
                torch.zeros(2, 2, 4, 4),
                torch.ones(2, 2, 4, 4),
            ]
        }
    }
    output = _stack_nested(value)
    assert output["modalities"]["sentinel1"].shape == (2, 2, 2, 4, 4)


def test_precomputed_backbone_validates_dimension():
    backbone = PrecomputedEmbeddingBackbone(out_channels=8)
    assert backbone(torch.zeros(2, 8, 4, 4))[0].shape == (2, 8, 4, 4)
    with pytest.raises(ValueError, match="Expected D=8"):
        backbone(torch.zeros(2, 7, 4, 4))
