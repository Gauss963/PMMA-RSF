"""Case-by-case PMMA runner with no plotting or animation side effects."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tatva.pmma.dynamics import (
    RunConfig,
    SimulationCheckpointed,
    run_simulation_dumped,
)
from tatva.pmma.config import PMMACaseConfig, RSFZoneConfig
from tatva.pmma.estimate import estimate_case_size
from tatva.pmma.model import (
    BlockSpec,
    FrictionReference,
    LoadingReference,
    Material,
    PMMAModelInput,
)
from tatva.pmma.mpi import get_mpi_context
from tatva.pmma.profiles import regularized_steady_friction


def make_case(config: PMMACaseConfig) -> PMMAModelInput:
    """Construct the Tatva model directly from a resolved PMMA case."""
    material = config.material
    materials = {
        name: Material(
            name=name,
            rho=material.density,
            E=material.young_modulus,
            nu=material.poisson_ratio,
        )
        for name in ("moving-block", "stationary-block")
    }
    return PMMAModelInput(
        moving=BlockSpec(
            name="moving-block",
            origin=config.moving.origin,
            dimensions=config.moving.dimensions,
            tag_prefix=1,
        ),
        stationary=BlockSpec(
            name="stationary-block",
            origin=config.stationary.origin,
            dimensions=config.stationary.dimensions,
            tag_prefix=2,
        ),
        materials=materials,
        friction=FrictionReference(
            mu_s=config.rsf.initial_friction,
            mu_k=config.rsf.target_middle_dynamic_friction,
            d_c=config.rsf.middle.characteristic_slip,
        ),
        simulation=LoadingReference(
            simulation_time=config.loading.shear_phase_time,
            time_factor=config.loading.shear_phase_time,
            normal_stress=config.loading.normal_stress_reference,
            rise_fraction=0.3,
            tau_k_start_fraction=1.0,
            normal_dir=0,
            slave_surface="stationary-block-back",
            master_surface="moving-block-front",
        ),
    )


def rate_state_profile_spec(config: PMMACaseConfig) -> dict[str, Any]:
    rsf = config.rsf

    def zone_payload(zone: RSFZoneConfig) -> dict[str, float]:
        return {
            "a": zone.direct_effect,
            "b": zone.state_effect,
            "dc": zone.characteristic_slip,
            "f0": (
                rsf.initial_friction
                if zone.reference_friction is None
                else zone.reference_friction
            ),
        }

    return {
        "initial_friction": rsf.initial_friction,
        "reference_velocity": rsf.reference_velocity,
        "reference_state": rsf.reference_state,
        "initial_steady_velocity": rsf.initial_steady_velocity,
        # Keep material profiles anchored to the original physical fault so a
        # geometry-only chamfer does not translate the RSF transition.
        "profile_length": config.moving.dimensions[1],
        "loading_length": rsf.loading_length,
        "leading_length": rsf.leading_length,
        "transition_length": rsf.transition_length,
        "loading_transition_length": rsf.loading_transition_length,
        "leading_transition_length": rsf.leading_transition_length,
        "loading": zone_payload(rsf.loading),
        "middle": zone_payload(rsf.middle),
        "leading": zone_payload(rsf.leading),
    }


def make_run_config(config: PMMACaseConfig) -> RunConfig:
    loading = config.loading
    numerics = config.numerics
    rsf = config.rsf
    return RunConfig(
        mesh_size=numerics.mesh_size,
        simulation_time=loading.shear_phase_time,
        cfl=numerics.cfl,
        dtype=numerics.dtype,
        operator_batch_size=numerics.operator_batch_size,
        moving_leading_chamfer_along_fault=(
            config.moving.leading_chamfer_along_fault
        ),
        moving_leading_chamfer_perpendicular=(
            config.moving.leading_chamfer_perpendicular
        ),
        normal_penalty=numerics.normal_penalty,
        tangential_penalty=numerics.tangential_penalty,
        contact_safety_factor=numerics.contact_safety_factor,
        time_step_override=numerics.time_step,
        normal_loading_mode="displacement",
        normal_displacement_override=loading.normal_displacement,
        shear_loading_mode="displacement",
        shear_displacement_k_override=loading.shear_displacement_initial,
        shear_displacement_s_override=loading.shear_displacement_final,
        normal_phase_time=loading.normal_phase_time,
        shear_phase_time=loading.shear_phase_time,
        normal_ramp_time=loading.normal_ramp_time,
        shear_ramp_time=loading.shear_ramp_time,
        shear_ramp_shape=loading.shear_ramp_shape,
        normal_relaxation_time=loading.effective_normal_relaxation_time,
        quasistatic_shear_fraction=loading.quasistatic_shear_fraction,
        quasistatic_shear_start_time=loading.quasistatic_shear_start_time,
        quasistatic_shear_ramp_time=loading.quasistatic_shear_ramp_time,
        stop_shear_loading_on_rupture=loading.stop_on_rupture,
        shear_loading_stop_slip=loading.stop_slip,
        shear_loading_stop_velocity=loading.stop_velocity,
        shear_loading_stop_min_y=loading.stop_min_y,
        shear_loading_stop_max_y=loading.stop_max_y,
        shear_loading_stop_coverage_fraction=loading.stop_coverage_fraction,
        lock_shear_edge_during_normal=loading.lock_shear_edge_during_normal,
        relax_tangential_contact_during_normal=(
            loading.relax_tangential_contact_during_normal
        ),
        friction_law="rate-state-regularized",
        rsf_reference_friction=rsf.initial_friction,
        rsf_direct_effect=rsf.middle.direct_effect,
        rsf_state_effect=rsf.middle.state_effect,
        rsf_reference_velocity=rsf.reference_velocity,
        rsf_reference_state=rsf.reference_state,
        rsf_characteristic_slip=rsf.middle.characteristic_slip,
        rsf_initial_state=rsf.middle.characteristic_slip
        / rsf.initial_steady_velocity,
        rsf_profile_spec=rate_state_profile_spec(config),
        normal_stress_override=loading.normal_stress_reference,
        dimension=2,
        thickness=0.0,
    )


RUN_SEQUENCE_PREFIX = "TS"
FIRST_RUN_NUMBER = 117
DEFAULT_MINIMUM_FREE_SPACE_RESERVE_BYTES = 50_000_000_000


def _minimum_free_space_reserve_bytes() -> int:
    raw_value = os.environ.get("PMMA_MINIMUM_FREE_SPACE_RESERVE_BYTES")
    if raw_value is None:
        return DEFAULT_MINIMUM_FREE_SPACE_RESERVE_BYTES
    reserve_bytes = int(raw_value)
    if reserve_bytes < 0:
        raise ValueError("PMMA_MINIMUM_FREE_SPACE_RESERVE_BYTES cannot be negative.")
    return reserve_bytes


def allocate_run_directory(root: Path) -> Path:
    """Atomically allocate the next label-free ``TS####`` run directory."""
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in root.iterdir():
        # Read both the new TS#### names and historical TS####_<label> names.
        match = re.fullmatch(rf"{RUN_SEQUENCE_PREFIX}(\d+)(?:_.*)?", path.name)
        if path.is_dir() and match:
            existing.append(int(match.group(1)))
    number = max(existing, default=FIRST_RUN_NUMBER - 1) + 1
    number = max(number, FIRST_RUN_NUMBER)
    while True:
        candidate = root / f"{RUN_SEQUENCE_PREFIX}{number:04d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            number += 1
            continue
        return candidate


def _validate_run_storage(
    path: Path,
    estimate: dict[str, Any],
    *,
    existing_dump_bytes: int = 0,
    reserve_bytes: int | None = None,
) -> dict[str, int]:
    """Require enough free space for the conservative remaining dump size."""
    if reserve_bytes is None:
        reserve_bytes = _minimum_free_space_reserve_bytes()
    estimated_uncompressed_bytes = int(estimate["estimated_uncompressed_bytes"])
    remaining_dump_bytes = max(
        estimated_uncompressed_bytes - int(existing_dump_bytes), 0
    )
    available_bytes = int(shutil.disk_usage(path).free)
    required_bytes = remaining_dump_bytes + int(reserve_bytes)
    report = {
        "available_bytes": available_bytes,
        "estimated_remaining_uncompressed_bytes": remaining_dump_bytes,
        "reserve_bytes": int(reserve_bytes),
        "required_bytes": required_bytes,
    }
    if available_bytes < required_bytes:
        raise OSError(
            "Insufficient free space for simulation dump: "
            f"{available_bytes / 1.0e12:.3f} TB available, "
            f"{required_bytes / 1.0e12:.3f} TB required "
            "(conservative remaining uncompressed dump plus reserve)."
        )
    return report


def preflight(config: PMMACaseConfig) -> dict[str, Any]:
    estimate = estimate_case_size(config)
    limit_bytes = config.output.maximum_dump_tb * 1.0e12
    estimated_dump_bytes = int(
        estimate["estimated_uncompressed_bytes"]
        * config.output.estimated_compression_ratio
    )
    estimate["maximum_dump_tb"] = config.output.maximum_dump_tb
    estimate["estimated_compression_ratio"] = (
        config.output.estimated_compression_ratio
    )
    estimate["estimated_dump_bytes"] = estimated_dump_bytes
    estimate["estimated_dump_tb"] = estimated_dump_bytes / 1.0e12
    estimate["within_dump_limit"] = estimated_dump_bytes <= limit_bytes
    estimate["animation_enabled"] = False
    estimate["animation_frames_written"] = False
    estimate["rsf_zones"] = {}
    for name in ("loading", "middle", "leading"):
        zone = getattr(config.rsf, name)
        reference_friction = (
            config.rsf.initial_friction
            if zone.reference_friction is None
            else zone.reference_friction
        )
        estimate["rsf_zones"][name] = {
            "a": zone.direct_effect,
            "b": zone.state_effect,
            "a_minus_b": zone.direct_effect - zone.state_effect,
            "dc_mm": zone.characteristic_slip,
            "reference_friction": reference_friction,
            "steady_mu_at_initial_velocity": regularized_steady_friction(
                velocity=config.rsf.initial_steady_velocity,
                reference_friction=reference_friction,
                direct_effect=zone.direct_effect,
                state_effect=zone.state_effect,
                reference_velocity=config.rsf.reference_velocity,
            ),
            "steady_mu_at_dynamic_calibration_velocity": (
                regularized_steady_friction(
                    velocity=config.rsf.dynamic_calibration_velocity,
                    reference_friction=reference_friction,
                    direct_effect=zone.direct_effect,
                    state_effect=zone.state_effect,
                    reference_velocity=config.rsf.reference_velocity,
                )
            ),
        }
    return estimate


def run_case(
    config: PMMACaseConfig,
    source_path: Path,
    *,
    run_root: Path,
    run_dir: Path | None = None,
    resume: bool = False,
    time_limit_seconds: float | None = None,
) -> Path:
    mpi_context = get_mpi_context()
    estimate = preflight(config)
    if not estimate["within_dump_limit"]:
        raise ValueError(
            "Configured dump estimate exceeds output.maximum_dump_tb: "
            f"{estimate['estimated_dump_tb']:.3f} TB > "
            f"{config.output.maximum_dump_tb:.3f} TB."
        )

    if resume and run_dir is None:
        raise ValueError("Resuming requires an explicit run_dir.")
    if time_limit_seconds is not None and time_limit_seconds <= 0.0:
        raise ValueError("time_limit_seconds must be positive.")

    run_root = run_root.expanduser().resolve()

    def prepare_public_run_directory() -> tuple[Path, dict[str, int]]:
        if run_dir is None:
            run_root.mkdir(parents=True, exist_ok=True)
            storage_report = _validate_run_storage(run_root, estimate)
            return allocate_run_directory(run_root), storage_report

        public_dir = run_dir.expanduser().resolve()
        if resume:
            if not public_dir.is_dir():
                raise FileNotFoundError(f"Run directory not found: {public_dir}")
        else:
            public_dir.mkdir(parents=True, exist_ok=True)
            conflicting = [
                path
                for name in ("input", "data", "stats", "status.json")
                if (path := public_dir / name).exists()
            ]
            if conflicting:
                raise FileExistsError(
                    "The requested run directory already contains simulation data: "
                    + ", ".join(str(path) for path in conflicting)
                )
        existing_dump = public_dir / "data" / "simulation.h5"
        storage_report = _validate_run_storage(
            public_dir,
            estimate,
            existing_dump_bytes=(
                existing_dump.stat().st_size
                if resume and existing_dump.exists()
                else 0
            ),
        )
        return public_dir, storage_report

    if mpi_context.is_root:
        public_run_dir, storage = prepare_public_run_directory()
        public_payload = (str(public_run_dir), storage)
    else:
        public_run_dir = Path()
        storage = {}
        public_payload = None
    if mpi_context.enabled:
        public_path, storage = mpi_context.comm.bcast(public_payload, root=0)
        public_run_dir = Path(public_path)

    if mpi_context.enabled and not mpi_context.is_root:
        job_id = os.environ.get("SLURM_JOB_ID", f"pid-{os.getppid()}")
        temporary_root = Path(
            os.environ.get("SLURM_TMPDIR", os.environ.get("TMPDIR", "/tmp"))
        )
        run_dir = temporary_root / f"pmma-rsf-{job_id}" / f"rank-{mpi_context.rank}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        if resume:
            for name in ("input", "data", "stats"):
                source = public_run_dir / name
                if source.exists():
                    shutil.copytree(source, run_dir / name)
            for name in ("status.json", "checkpoint.npz"):
                source = public_run_dir / name
                if source.exists():
                    shutil.copy2(source, run_dir / name)
        existing_dump = run_dir / "data" / "simulation.h5"
        _validate_run_storage(
            run_dir,
            estimate,
            existing_dump_bytes=(
                existing_dump.stat().st_size
                if resume and existing_dump.exists()
                else 0
            ),
            reserve_bytes=0,
        )
    else:
        run_dir = public_run_dir
    estimate["storage_preflight"] = storage
    estimate["mpi_ranks"] = mpi_context.size
    data_dir = run_dir / "data"
    stats_dir = run_dir / "stats"
    input_dir = run_dir / "input"
    logs_dir = run_dir / "logs"
    for directory in (data_dir, stats_dir, input_dir, logs_dir):
        directory.mkdir(exist_ok=True)
    resolved_path = input_dir / "resolved_case.json"
    resolved_config = json.loads(json.dumps(config.as_dict()))
    if resume:
        if not resolved_path.exists():
            raise FileNotFoundError(f"Resolved case not found: {resolved_path}")
        existing_config = json.loads(resolved_path.read_text(encoding="utf-8"))
        if existing_config != resolved_config:
            raise ValueError("Resume input does not match the original resolved case.")
        with (input_dir / "resume_commands.txt").open("a", encoding="utf-8") as stream:
            stream.write(" ".join(sys.argv) + "\n")
    else:
        shutil.copy2(source_path, input_dir / "case.toml")
        resolved_path.write_text(
            json.dumps(resolved_config, indent=2), encoding="utf-8"
        )
        (stats_dir / "preflight.json").write_text(
            json.dumps(estimate, indent=2), encoding="utf-8"
        )
        (input_dir / "command.txt").write_text(
            " ".join(sys.argv) + "\n", encoding="utf-8"
        )
    status_path = run_dir / "status.json"
    previous_status = {}
    if resume and status_path.exists():
        previous_status = json.loads(status_path.read_text(encoding="utf-8"))
    now_utc = datetime.now(timezone.utc).isoformat()
    status = {
        "status": "running",
        "started_utc": previous_status.get("started_utc", now_utc),
        "resumed_utc": now_utc if resume else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "animation": "disabled by PMMA HPC runner",
        "resume_count": int(previous_status.get("resume_count", 0)) + int(resume),
        "time_limit_seconds": time_limit_seconds,
        "mpi_rank": mpi_context.rank,
        "mpi_ranks": mpi_context.size,
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    deadline = (
        None if time_limit_seconds is None else time.monotonic() + time_limit_seconds
    )
    runner_started_monotonic = time.monotonic()
    try:
        result = run_simulation_dumped(
            make_case(config),
            make_run_config(config),
            data_dir / "simulation.h5",
            frames_per_phase=config.output.bulk_normal_frames,
            shear_frames_per_phase=config.output.bulk_shear_frames,
            interface_frames_per_phase=config.output.interface_normal_frames,
            shear_interface_frames_per_phase=config.output.interface_shear_frames,
            compression=config.output.compression,
            include_initial_frame=config.output.include_initial_frame,
            store_bulk_strain=config.output.store_bulk_strain,
            store_bulk_velocity=config.output.store_bulk_velocity,
            checkpoint_path=run_dir / "checkpoint.npz",
            checkpoint_interval_seconds=(
                60.0 * config.output.checkpoint_interval_minutes
            ),
            checkpoint_deadline_monotonic=deadline,
            resume=resume,
        )
        summary = {
            "summary": result["summary"],
            "preflight": estimate,
            "run_dir": str(run_dir),
            "data_path": str(data_dir / "simulation.h5"),
            "animation": None,
        }
        (stats_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        status.update(
            status="complete",
            completed_utc=datetime.now(timezone.utc).isoformat(),
        )
    except SimulationCheckpointed as exc:
        status.update(
            status="checkpointed",
            checkpointed_utc=datetime.now(timezone.utc).isoformat(),
            checkpoint=str(run_dir / "checkpoint.npz"),
            message=str(exc),
        )
    except BaseException as exc:
        status.update(
            status="failed",
            failed_utc=datetime.now(timezone.utc).isoformat(),
            error=f"{type(exc).__name__}: {exc}",
        )
        (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["runner_elapsed_seconds"] = (
            time.monotonic() - runner_started_monotonic
        )
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if mpi_context.enabled:
        mpi_context.comm.Barrier()
    if (
        mpi_context.enabled
        and not mpi_context.is_root
        and status["status"] in {"complete", "checkpointed"}
    ):
        shutil.rmtree(run_dir)
    return public_run_dir
