from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_normal_phase_end_normal_stress_profiles import (  # noqa: E402
    last_normal_frame,
    plot_normal_phase_end_normal_stress_profiles,
    profile_at_nearest_element_columns,
)


def _structured_mesh() -> tuple[np.ndarray, np.ndarray]:
    x_values = np.arange(0.0, 2.5, 0.5)
    y_values = np.arange(0.0, 3.0, 1.0)
    xx, yy = np.meshgrid(x_values, y_values, indexing="xy")
    coords = np.column_stack((xx.ravel(), yy.ravel()))
    nx = x_values.size
    elements = []
    for iy in range(y_values.size - 1):
        for ix in range(x_values.size - 1):
            n00 = iy * nx + ix
            elements.append([n00, n00 + 1, n00 + 1 + nx, n00 + nx])
    return coords, np.asarray(elements, dtype=np.int64)


def test_last_normal_frame_rejects_missing_phase():
    assert last_normal_frame(np.asarray([1, 1, 2, 2])) == 1
    with pytest.raises(ValueError, match="no normal-phase"):
        last_normal_frame(np.asarray([2, 2]))


def test_profile_at_nearest_columns_averages_equidistant_columns():
    coords, elements = _structured_mesh()
    centers = coords[elements].mean(axis=1)
    values = 10.0 * centers[:, 0] + centers[:, 1]

    y, profile, columns = profile_at_nearest_element_columns(centers, values, 1.0)

    np.testing.assert_allclose(columns, [0.75, 1.25])
    np.testing.assert_allclose(y, [0.5, 1.5])
    np.testing.assert_allclose(profile, [10.5, 11.5])


def test_plot_uses_last_normal_frame_and_writes_stats(tmp_path):
    coords, elements = _structured_mesh()
    centers = coords[elements].mean(axis=1)
    input_path = tmp_path / "run" / "data" / "simulation.h5"
    output_path = tmp_path / "run" / "Plot" / "normal_phase_end_normal_stress_profiles.pdf"
    stats_dir = tmp_path / "run" / "stats"
    input_path.parent.mkdir(parents=True)

    sigma = np.zeros((3, elements.shape[0], 2, 2), dtype=np.float32)
    sigma[0, :, 0, 0] = -1.0
    sigma[1, :, 0, 0] = -(20.0 + 10.0 * centers[:, 0] + centers[:, 1])
    sigma[2, :, 0, 0] = -999.0
    with h5py.File(input_path, "w") as h5:
        h5.attrs["dt"] = 1.0e-3
        h5.attrs["normal_stress"] = 16.0
        h5.attrs["normal_loading_mode"] = "displacement"
        h5.attrs["normal_displacement_loading_fraction"] = 0.8
        h5.attrs["normal_displacement_leading_fraction"] = 1.2
        h5.create_dataset("phase_id", data=np.asarray([1, 1, 2]))
        h5.create_dataset("step_id", data=np.asarray([1, 2, 1]))
        h5.create_dataset("history", data=np.zeros((3, 2)))
        moving = h5.create_group("moving")
        moving.create_dataset("coords", data=coords)
        moving.create_dataset("elements", data=elements)
        moving.create_dataset("stress", data=sigma)

    result = plot_normal_phase_end_normal_stress_profiles(
        input_path,
        output_path,
        stats_dir,
        distance_from_fault_mm=1.0,
        dpi=72,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert result["frame_index"] == 1
    assert result["time_ms"] == pytest.approx(2.0)
    assert result["stress_component"] == "-sigma_xx"
    assert result["normal_loading_mode"] == "displacement"
    assert result["loading_boundary_element_center_x_mm"] == pytest.approx([0.25])
    assert result["near_fault_element_center_x_mm"] == pytest.approx([0.75, 1.25])

    csv_path = stats_dir / "normal_phase_end_normal_stress_profiles.csv"
    json_path = stats_dir / "normal_phase_end_normal_stress_profiles.json"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert float(rows[0]["loading_boundary_compressive_normal_stress_mpa"]) == pytest.approx(23.0)
    near_fault_value = float(
        rows[0]["moving_1mm_from_fault_compressive_normal_stress_mpa"]
    )
    assert near_fault_value == pytest.approx(30.5)
    assert payload["normal_displacement_loading_fraction"] == pytest.approx(0.8)
    assert payload["normal_displacement_leading_fraction"] == pytest.approx(1.2)
