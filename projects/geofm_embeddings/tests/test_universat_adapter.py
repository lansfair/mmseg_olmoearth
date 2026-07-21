from __future__ import annotations

import torch
from torch import nn

from projects.geofm_embeddings.geofm_embeddings.adapters.universat import (
    S2_OLMOEARTH_TO_UNIVERSAT,
    UniverSatAdapter,
    _timestamps_to_reference_days,
)


class FakeUniverSat(nn.Module):
    def __init__(self, output_grid: int = 2, out_channels: int = 4) -> None:
        super().__init__()
        self.output_grid = output_grid
        self.out_channels = out_channels
        self.last_inputs = None

    def encode(self, inputs, patch_size, output_grid):
        self.last_inputs = inputs
        assert output_grid == self.output_grid
        batch_size = inputs["s2"].shape[0]
        values = torch.arange(
            batch_size * output_grid * output_grid * self.out_channels,
            dtype=inputs["s2"].dtype,
            device=inputs["s2"].device,
        )
        return values.reshape(
            batch_size, output_grid * output_grid, self.out_channels
        ), {}


def _inputs():
    s2 = torch.arange(2 * 3 * 12 * 8 * 8, dtype=torch.float32).reshape(
        2, 3, 12, 8, 8
    )
    timestamps = torch.tensor(
        [
            [[1, 0, 2018], [2, 0, 2018], [1, 1, 2018]],
            [[31, 11, 2018], [1, 0, 2019], [2, 0, 2019]],
        ]
    )
    return {
        "modalities": {"sentinel2_l2a": s2},
        "timestamps": {"sentinel2_l2a": timestamps},
    }


def test_universat_timestamp_conversion_uses_zero_based_months():
    timestamps = torch.tensor(
        [[[1, 0, 2018], [1, 1, 2018], [1, 0, 2019]]]
    )
    assert _timestamps_to_reference_days(timestamps).tolist() == [
        [0, 31, 365]
    ]


def test_universat_adapter_reorders_s2_and_returns_dense_layout():
    model = FakeUniverSat()
    adapter = UniverSatAdapter(
        model=model,
        output_grid=2,
        out_channels=4,
        freeze=False,
    )
    result = adapter.extract(_inputs(), mode="dense")
    assert result.tensor.shape == (2, 4, 2, 2)
    expected = _inputs()["modalities"]["sentinel2_l2a"][
        :, :, S2_OLMOEARTH_TO_UNIVERSAT
    ]
    assert torch.equal(model.last_inputs["s2"], expected)
    assert model.last_inputs["s2_dates"].tolist() == [
        [0, 1, 31],
        [364, 365, 366],
    ]


def test_universat_adapter_global_pooling():
    adapter = UniverSatAdapter(
        model=FakeUniverSat(),
        output_grid=2,
        out_channels=4,
        freeze=False,
    )
    result = adapter.extract(_inputs(), mode="global")
    assert result.tensor.shape == (2, 4)
    dense = torch.arange(2 * 2 * 2 * 4, dtype=torch.float32).reshape(
        2, 2, 2, 4
    ).permute(0, 3, 1, 2)
    assert torch.equal(result.tensor, dense.mean(dim=(-2, -1)))
