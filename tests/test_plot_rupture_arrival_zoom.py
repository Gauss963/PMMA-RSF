import json
import sys
from pathlib import Path

import h5py
import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from plot_rupture_arrival_zoom import (  # noqa: E402
    first_threshold_crossing,
    local_speeds,
    plot_rupture_arrival_zoom,
)


def _write_synthetic_lsw_dump(path, speed_m_per_s=900.0, creep_from_mm=None):
    """Front leaves y=0 at 2 ms; slip ramps at 2 m/s behind it; loading stops at 2.5 ms."""
    dt = 1.0e-6
    pressure_steps = 100
    y = np.arange(0.0, 501.0, 5.0)
    shear_steps = np.arange(1, 6001, 2)
    time_ms = shear_steps * dt * 1e3
    arrival = 2.0 + y / speed_m_per_s
    elapsed = np.clip(time_ms[:, None] - arrival[None, :], 0.0, None)
    slip = 2.0 * elapsed  # mm, from a 2 m/s (= 2 mm/ms) slip rate
    rate = np.where(elapsed > 0.0, 2000.0, 0.0)
    critical_slip = np.full(y.shape, 0.01)
    columns = ["time", "shear_loading_stopped"]
    with h5py.File(path, "w") as h5:
        h5.attrs.update(dt=dt, pressure_steps=pressure_steps, friction_law="slip-weakening",
                        young_modulus=7662.0, poisson_ratio=0.2, density=1.148e-9)
        coords = np.stack([np.full(y.shape, 200.0), y], axis=1)
        h5["moving/coords"] = coords
        interface = h5.create_group("interface")
        interface["master_nodes"] = np.arange(y.size)
        interface["critical_slip_profile"] = critical_slip
        weight = np.zeros_like(y) if creep_from_mm is None else (y >= creep_from_mm).astype(float)
        interface["creep_weight_profile"] = weight
        # One normal frame followed by the shear frames.
        h5["phase_id"] = np.concatenate([[1], np.full(shear_steps.size, 2)])
        h5["step_id"] = np.concatenate([[pressure_steps], shear_steps])
        interface["cumulative_slip"] = np.vstack([np.zeros(y.size), slip])
        interface["slip_rate"] = np.vstack([np.zeros(y.size), rate])
        mu = np.where(slip >= critical_slip, 0.45, 0.8)
        interface["friction_coefficient"] = np.vstack([np.full(y.size, 0.8), mu])
        history = np.zeros((shear_steps.size + 1, len(columns)))
        history[1:, 1] = time_ms >= 2.5
        h5["history"] = history
        h5["history_columns"] = np.asarray(columns, dtype="S32")
    return y


def test_first_threshold_crossing_returns_first_saved_time_or_nan():
    values = np.array([[0.0, 0.0], [50.0, -200.0], [150.0, 0.0]])
    arrivals = first_threshold_crossing(values, np.array([0.0, 1.0, 2.0]), 100.0)
    np.testing.assert_allclose(arrivals, [2.0, 1.0])
    assert np.isnan(first_threshold_crossing(values, np.array([0.0, 1.0, 2.0]), 1.0e3)).all()


def test_local_speeds_report_signed_bin_speed():
    y = np.arange(0.0, 100.0, 1.0)
    arrival = np.where(y < 50.0, 1.0 - y / 500.0, y / 1000.0)
    speeds = [row["speed_m_per_s"] for row in local_speeds(y, arrival, width_mm=50.0)]
    np.testing.assert_allclose(speeds, [-500.0, 1000.0], rtol=1e-9)


def test_zoom_recovers_front_speed_stop_and_writes_outputs(tmp_path):
    dump = tmp_path / "run" / "data" / "simulation.h5"
    dump.parent.mkdir(parents=True)
    _write_synthetic_lsw_dump(dump, speed_m_per_s=900.0)
    output = tmp_path / "run" / "Plot" / "rupture_arrival_zoom.pdf"
    stats = tmp_path / "run" / "stats"

    summary = plot_rupture_arrival_zoom(dump, output, stats, dpi=80)

    assert output.is_file() and output.with_suffix(".png").is_file()
    saved = json.loads((stats / "rupture_arrival_zoom.json").read_text())
    assert saved["fit_interval_mm"] == [200.0, 500.0]
    assert saved["leading_creep_start_mm"] is None
    assert abs(saved["shear_loading_stop_first_saved_ms"] - 2.5) <= 1e-3 + 1e-9
    for name in ("slip_dc", "rate_100", "rate_500", "rate_1000"):
        arrival = saved["arrivals"][name]
        assert arrival["fit_available"]
        # Saved frames are 2 us apart, so allow one frame of bias across 300 mm.
        assert abs(arrival["speed_m_per_s"] - 900.0) < 10.0
        assert arrival["first_position_mm"] == 0.0
    window = summary["window_ms"]
    assert window[0] < 2.0 < 2.0 + 500.0 / 900.0 < window[1]


def test_zoom_fit_stops_before_lsw_creep_zone(tmp_path):
    dump = tmp_path / "simulation.h5"
    _write_synthetic_lsw_dump(dump, creep_from_mm=480.0)

    summary = plot_rupture_arrival_zoom(dump, tmp_path / "zoom.pdf", tmp_path, dpi=80)

    assert summary["leading_creep_start_mm"] == 480.0
    assert summary["fit_interval_mm"] == [200.0, 475.0]
