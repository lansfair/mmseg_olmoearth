from __future__ import annotations

import torch

from projects.geofm_embeddings.geofm_embeddings.adapters.copernicusfm import (
    SPECTRAL_METADATA,
    _clean_state_dict,
)
from projects.geofm_embeddings.geofm_embeddings.adapters.dinov3 import (
    DINOv3Adapter,
)
from projects.geofm_embeddings.geofm_embeddings.adapters.tessera import (
    TESSERAAdapter,
)
from projects.geofm_embeddings.geofm_embeddings.adapters.official_wrapper import (
    OFFICIAL_WRAPPER_PRESETS,
)


def test_dinov3_selects_sentinel2_rgb_in_rgb_order():
    value = torch.arange(12.0)[None, None, :, None, None]
    rgb = DINOv3Adapter.select_rgb(value, "sentinel2_l2a")
    assert rgb.flatten().tolist() == [2.0, 1.0, 0.0]


def test_dinov3_preserves_preselected_rgb():
    value = torch.randn(2, 4, 3, 8, 8)
    assert DINOv3Adapter.select_rgb(value, "sentinel2_l2a") is value


def test_copernicus_spectral_metadata_matches_input_bands():
    for metadata in SPECTRAL_METADATA.values():
        assert set(metadata["bands"]) == set(metadata["wavelength"])
        assert set(metadata["bands"]) == set(metadata["bandwidth"])


def test_copernicus_state_dict_prefix_cleanup():
    state = _clean_state_dict(
        {"model": {"module.model.blocks.0.weight": torch.ones(1)}}
    )
    assert set(state) == {"blocks.0.weight"}


def test_tessera_day_of_year_with_zero_based_months():
    timestamps = torch.tensor(
        [
            [1, 0, 2024],
            [1, 1, 2024],
            [1, 2, 2024],
        ]
    )
    doy = TESSERAAdapter.calculate_day_of_year(timestamps, month_base=0)
    assert doy.tolist() == [1, 32, 61]


def test_official_wrapper_presets_cover_remaining_families():
    assert set(OFFICIAL_WRAPPER_PRESETS) == {
        "anysat",
        "clay",
        "croma",
        "galileo",
        "panopticon",
        "presto",
        "prithviv2",
        "satlas",
        "terramind",
    }
    assert OFFICIAL_WRAPPER_PRESETS["panopticon"]["dense_method"] == (
        "forward_features"
    )
