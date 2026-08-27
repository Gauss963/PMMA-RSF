"""Typed TOML configuration for PMMA rate-and-state simulations."""

from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BlockConfig:
    origin: tuple[float, float]
    dimensions: tuple[float, float]


@dataclass(frozen=True)
class MaterialConfig:
    density: float
    young_modulus: float
    poisson_ratio: float


@dataclass(frozen=True)
class LoadingConfig:
    normal_stress_reference: float
    normal_displacement: float
    normal_phase_time: float
    normal_ramp_time: float
    shear_displacement_initial: float
    shear_displacement_final: float
    shear_phase_time: float
    shear_ramp_time: float
    shear_ramp_shape: str
    stop_on_rupture: bool
    stop_slip: float
    stop_velocity: float | None
    stop_min_y: float
    stop_max_y: float
    stop_coverage_fraction: float | None
    lock_shear_edge_during_normal: bool
    relax_tangential_contact_during_normal: bool
    quasistatic_shear_fraction: float
    quasistatic_shear_start_time: float
    quasistatic_shear_ramp_time: float
    normal_relaxation_time: float | None
    quasistatic_damping_time: float | None

    @property
    def effective_normal_relaxation_time(self) -> float | None:
        if self.normal_relaxation_time is not None:
            return self.normal_relaxation_time
        return self.quasistatic_damping_time


@dataclass(frozen=True)
class NumericsConfig:
    mesh_size: float
    cfl: float
    dtype: str
    time_step: float | None = None
    operator_batch_size: int | None = None
    normal_penalty: float | None = None
    tangential_penalty: float | None = None
    contact_safety_factor: float = 0.25


@dataclass(frozen=True)
class OutputConfig:
    bulk_normal_frames: int
    bulk_shear_frames: int
    interface_normal_frames: int
    interface_shear_frames: int
    compression: str
    estimated_compression_ratio: float
    include_initial_frame: bool
    maximum_dump_tb: float
    checkpoint_interval_minutes: float
    store_bulk_strain: bool = True
    store_bulk_velocity: bool = True


@dataclass(frozen=True)
class RSFZoneConfig:
    direct_effect: float
    state_effect: float
    characteristic_slip: float
    reference_friction: float | None = None


@dataclass(frozen=True)
class RSFConfig:
    initial_friction: float
    reference_velocity: float
    reference_state: float
    initial_steady_velocity: float
    dynamic_calibration_velocity: float
    target_middle_dynamic_friction: float
    loading_length: float
    leading_length: float
    transition_length: float
    loading: RSFZoneConfig
    middle: RSFZoneConfig
    leading: RSFZoneConfig


@dataclass(frozen=True)
class PMMACaseConfig:
    name: str
    run_root: str
    moving: BlockConfig
    stationary: BlockConfig
    material: MaterialConfig
    loading: LoadingConfig
    numerics: NumericsConfig
    output: OutputConfig
    rsf: RSFConfig

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuple2(values: list[float] | tuple[float, ...], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    return float(values[0]), float(values[1])


def _zone(payload: dict[str, Any], name: str) -> RSFZoneConfig:
    try:
        zone = RSFZoneConfig(
            direct_effect=float(payload["a"]),
            state_effect=float(payload["b"]),
            characteristic_slip=float(payload["dc"]),
            reference_friction=(
                None
                if "f0" not in payload
                else float(payload["f0"])
            ),
        )
    except KeyError as exc:
        raise ValueError(f"Missing rsf.{name}.{exc.args[0]} in the case file.") from exc
    if zone.direct_effect <= 0.0 or zone.state_effect < 0.0:
        raise ValueError(f"rsf.{name} requires a > 0 and b >= 0.")
    if zone.characteristic_slip <= 0.0:
        raise ValueError(f"rsf.{name}.dc must be positive.")
    if zone.reference_friction is not None and zone.reference_friction <= 0.0:
        raise ValueError(f"rsf.{name}.f0 must be positive.")
    return zone


def load_case_config(path: str | Path) -> PMMACaseConfig:
    """Load and validate one self-contained PMMA TOML case."""
    source = Path(path).expanduser().resolve()
    with source.open("rb") as stream:
        payload = tomllib.load(stream)

    case_data = payload["case"]
    geometry = payload["geometry"]
    material = payload["material"]
    loading = payload["loading"]
    numerics = payload["numerics"]
    output = payload["output"]
    rsf = payload["rsf"]

    config = PMMACaseConfig(
        name=str(case_data["name"]),
        run_root=str(case_data.get("run_root", "runs")),
        moving=BlockConfig(
            origin=_tuple2(geometry["moving"]["origin"], "geometry.moving.origin"),
            dimensions=_tuple2(
                geometry["moving"]["dimensions"], "geometry.moving.dimensions"
            ),
        ),
        stationary=BlockConfig(
            origin=_tuple2(
                geometry["stationary"]["origin"], "geometry.stationary.origin"
            ),
            dimensions=_tuple2(
                geometry["stationary"]["dimensions"],
                "geometry.stationary.dimensions",
            ),
        ),
        material=MaterialConfig(
            density=float(material["density"]),
            young_modulus=float(material["young_modulus"]),
            poisson_ratio=float(material["poisson_ratio"]),
        ),
        loading=LoadingConfig(
            normal_stress_reference=float(loading["normal_stress_reference"]),
            normal_displacement=float(loading["normal_displacement"]),
            normal_phase_time=float(loading["normal_phase_time"]),
            normal_ramp_time=float(loading["normal_ramp_time"]),
            shear_displacement_initial=float(loading["shear_displacement_initial"]),
            shear_displacement_final=float(loading["shear_displacement_final"]),
            shear_phase_time=float(loading["shear_phase_time"]),
            shear_ramp_time=float(loading["shear_ramp_time"]),
            shear_ramp_shape=str(loading["shear_ramp_shape"]),
            stop_on_rupture=bool(loading["stop_on_rupture"]),
            stop_slip=float(loading["stop_slip"]),
            stop_velocity=(
                None
                if "stop_velocity" not in loading
                else float(loading["stop_velocity"])
            ),
            stop_min_y=float(loading["stop_min_y"]),
            stop_max_y=float(loading["stop_max_y"]),
            stop_coverage_fraction=(
                None
                if "stop_coverage_fraction" not in loading
                else float(loading["stop_coverage_fraction"])
            ),
            lock_shear_edge_during_normal=bool(
                loading["lock_shear_edge_during_normal"]
            ),
            relax_tangential_contact_during_normal=bool(
                loading.get("relax_tangential_contact_during_normal", False)
            ),
            quasistatic_shear_fraction=float(
                loading.get("quasistatic_shear_fraction", 0.0)
            ),
            quasistatic_shear_start_time=float(
                loading.get("quasistatic_shear_start_time", 0.0)
            ),
            quasistatic_shear_ramp_time=float(
                loading.get("quasistatic_shear_ramp_time", 0.0)
            ),
            normal_relaxation_time=(
                None
                if "normal_relaxation_time" not in loading
                else float(loading["normal_relaxation_time"])
            ),
            quasistatic_damping_time=(
                None
                if "quasistatic_damping_time" not in loading
                else float(loading["quasistatic_damping_time"])
            ),
        ),
        numerics=NumericsConfig(
            mesh_size=float(numerics["mesh_size"]),
            cfl=float(numerics["cfl"]),
            dtype=str(numerics["dtype"]),
            time_step=(
                None
                if "time_step" not in numerics
                else float(numerics["time_step"])
            ),
            operator_batch_size=(
                None
                if "operator_batch_size" not in numerics
                else int(numerics["operator_batch_size"])
            ),
            normal_penalty=(
                None
                if "normal_penalty" not in numerics
                else float(numerics["normal_penalty"])
            ),
            tangential_penalty=(
                None
                if "tangential_penalty" not in numerics
                else float(numerics["tangential_penalty"])
            ),
            contact_safety_factor=float(
                numerics.get("contact_safety_factor", 0.25)
            ),
        ),
        output=OutputConfig(
            bulk_normal_frames=int(output["bulk_normal_frames"]),
            bulk_shear_frames=int(output["bulk_shear_frames"]),
            interface_normal_frames=int(output["interface_normal_frames"]),
            interface_shear_frames=int(output["interface_shear_frames"]),
            compression=str(output["compression"]),
            estimated_compression_ratio=float(
                output.get("estimated_compression_ratio", 1.0)
            ),
            include_initial_frame=bool(output["include_initial_frame"]),
            maximum_dump_tb=float(output["maximum_dump_tb"]),
            checkpoint_interval_minutes=float(
                output.get("checkpoint_interval_minutes", 20.0)
            ),
            store_bulk_strain=bool(output.get("store_bulk_strain", True)),
            store_bulk_velocity=bool(output.get("store_bulk_velocity", True)),
        ),
        rsf=RSFConfig(
            initial_friction=float(rsf["initial_friction"]),
            reference_velocity=float(rsf["reference_velocity"]),
            reference_state=float(rsf["reference_state"]),
            initial_steady_velocity=float(rsf["initial_steady_velocity"]),
            dynamic_calibration_velocity=float(rsf["dynamic_calibration_velocity"]),
            target_middle_dynamic_friction=float(
                rsf["target_middle_dynamic_friction"]
            ),
            loading_length=float(rsf["loading_length"]),
            leading_length=float(rsf["leading_length"]),
            transition_length=float(rsf["transition_length"]),
            loading=_zone(rsf["loading"], "loading"),
            middle=_zone(rsf["middle"], "middle"),
            leading=_zone(rsf["leading"], "leading"),
        ),
    )
    _validate(config)
    return config


def _validate(config: PMMACaseConfig) -> None:
    if config.numerics.mesh_size <= 0.0:
        raise ValueError("numerics.mesh_size must be positive.")
    if not 0.0 < config.numerics.cfl <= 1.0:
        raise ValueError("numerics.cfl must be in the interval (0, 1].")
    if config.numerics.time_step is not None and config.numerics.time_step <= 0.0:
        raise ValueError("numerics.time_step must be positive when specified.")
    if not 0.0 < config.numerics.contact_safety_factor <= 2.0:
        raise ValueError(
            "numerics.contact_safety_factor must be in the interval (0, 2]."
        )
    if config.numerics.dtype not in {"float32", "float64"}:
        raise ValueError("numerics.dtype must be float32 or float64.")
    if (
        config.numerics.operator_batch_size is not None
        and config.numerics.operator_batch_size <= 0
    ):
        raise ValueError("numerics.operator_batch_size must be positive.")
    if config.output.compression not in {"lzf", "gzip"}:
        raise ValueError("output.compression must be lzf or gzip.")
    if not 0.0 < config.output.estimated_compression_ratio <= 1.0:
        raise ValueError(
            "output.estimated_compression_ratio must be in the interval (0, 1]."
        )
    frame_values = (
        config.output.bulk_normal_frames,
        config.output.bulk_shear_frames,
        config.output.interface_normal_frames,
        config.output.interface_shear_frames,
    )
    if any(value <= 0 for value in frame_values):
        raise ValueError("All output frame counts must be positive.")
    if config.output.interface_normal_frames < config.output.bulk_normal_frames:
        raise ValueError("interface_normal_frames cannot be below bulk_normal_frames.")
    if config.output.interface_shear_frames < config.output.bulk_shear_frames:
        raise ValueError("interface_shear_frames cannot be below bulk_shear_frames.")
    if config.output.maximum_dump_tb <= 0.0:
        raise ValueError("output.maximum_dump_tb must be positive.")
    if config.output.checkpoint_interval_minutes <= 0.0:
        raise ValueError("output.checkpoint_interval_minutes must be positive.")
    if config.loading.shear_ramp_time > config.loading.shear_phase_time:
        raise ValueError("shear_ramp_time cannot exceed shear_phase_time.")
    if config.loading.normal_ramp_time > config.loading.normal_phase_time:
        raise ValueError("normal_ramp_time cannot exceed normal_phase_time.")
    if config.loading.stop_slip <= 0.0:
        raise ValueError("loading.stop_slip must be positive.")
    if (
        config.loading.stop_velocity is not None
        and config.loading.stop_velocity <= 0.0
    ):
        raise ValueError("loading.stop_velocity must be positive.")
    if config.loading.stop_min_y > config.loading.stop_max_y:
        raise ValueError("loading.stop_min_y cannot exceed stop_max_y.")
    if (
        config.loading.stop_coverage_fraction is not None
        and not 0.0 < config.loading.stop_coverage_fraction <= 1.0
    ):
        raise ValueError("loading.stop_coverage_fraction must be in (0, 1].")
    if not 0.0 <= config.loading.quasistatic_shear_fraction < 1.0:
        raise ValueError("quasistatic_shear_fraction must be in [0, 1).")
    relaxation_time = config.loading.effective_normal_relaxation_time
    if (
        config.loading.normal_relaxation_time is not None
        and config.loading.quasistatic_damping_time is not None
        and not math.isclose(
            config.loading.normal_relaxation_time,
            config.loading.quasistatic_damping_time,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        )
    ):
        raise ValueError(
            "normal_relaxation_time and quasistatic_damping_time must match "
            "when both are provided."
        )
    if relaxation_time is not None and relaxation_time <= 0.0:
        raise ValueError("normal_relaxation_time must be positive when enabled.")
    if config.loading.quasistatic_shear_fraction > 0.0:
        if config.loading.quasistatic_shear_start_time < 0.0:
            raise ValueError("quasistatic_shear_start_time cannot be negative.")
        if config.loading.quasistatic_shear_ramp_time <= 0.0:
            raise ValueError(
                "quasistatic_shear_ramp_time must be positive when enabled."
            )
        if (
            config.loading.quasistatic_shear_start_time
            + config.loading.quasistatic_shear_ramp_time
            > config.loading.normal_phase_time
        ):
            raise ValueError(
                "quasistatic shear ramp must finish within normal_phase_time."
            )
        if relaxation_time is None:
            raise ValueError(
                "normal_relaxation_time must be provided when quasistatic shear "
                "preloading is enabled."
            )
    fault_length = config.moving.dimensions[1]
    occupied = (
        config.rsf.loading_length
        + config.rsf.leading_length
        + 2.0 * config.rsf.transition_length
    )
    if occupied >= fault_length:
        raise ValueError("RSF end zones and transitions leave no middle fault segment.")
    if config.rsf.initial_steady_velocity <= 0.0:
        raise ValueError("rsf.initial_steady_velocity must be positive.")
    if config.rsf.reference_velocity <= 0.0 or config.rsf.reference_state <= 0.0:
        raise ValueError("RSF reference velocity and state must be positive.")
