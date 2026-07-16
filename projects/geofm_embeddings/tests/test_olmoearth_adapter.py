from __future__ import annotations

import pytest
import torch

from projects.geofm_embeddings.geofm_embeddings.adapters.olmoearth import (
    OlmoEarthAdapter,
)


def test_olmoearth_coerce_canonical_layout():
    value = torch.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)
    output = OlmoEarthAdapter._coerce_modality_tensor(
        value,
        modality="test",
        num_bands=4,
        num_timesteps=3,
    )
    assert output.shape == (2, 5, 6, 3, 4)
    assert output[0, 0, 0, 0, 0] == value[0, 0, 0, 0, 0]


def test_olmoearth_coerce_legacy_layout():
    value = torch.arange(2 * 12 * 5 * 6).reshape(2, 12, 5, 6)
    output = OlmoEarthAdapter._coerce_modality_tensor(
        value,
        modality="test",
        num_bands=4,
        num_timesteps=3,
    )
    assert output.shape == (2, 5, 6, 3, 4)


def test_olmoearth_coerce_rejects_wrong_channels():
    with pytest.raises(ValueError, match=r"expected C\*T=12"):
        OlmoEarthAdapter._coerce_modality_tensor(
            torch.zeros(2, 11, 5, 6),
            modality="test",
            num_bands=4,
            num_timesteps=3,
        )
