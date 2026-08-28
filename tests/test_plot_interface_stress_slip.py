from __future__ import annotations

import pytest

from scripts.plot_interface_stress_slip import subplot_grid_shape


@pytest.mark.parametrize(
    ("station_count", "expected"),
    [
        (1, (1, 1)),
        (4, (2, 2)),
        (5, (3, 2)),
        (8, (4, 2)),
    ],
)
def test_subplot_grid_shape_adapts_to_station_count(
    station_count: int,
    expected: tuple[int, int],
) -> None:
    assert subplot_grid_shape(station_count) == expected


def test_subplot_grid_shape_rejects_empty_station_list() -> None:
    with pytest.raises(ValueError, match="At least one station"):
        subplot_grid_shape(0)
