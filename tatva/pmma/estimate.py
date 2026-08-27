"""Mesh and HDF5 capacity estimates for structured 2-D PMMA cases."""

from __future__ import annotations

import math
from typing import Any

from tatva.pmma.config import PMMACaseConfig


TPV102_REFERENCE_DOFS = 51_252_333

# Structured 2-D blocks are meshed with one Quad4 per grid cell; see
# create_structured_quad_block in tatva.pmma.dynamics.
ELEMENTS_PER_CELL = 1
NODES_PER_ELEMENT = 4


def _axis_count_and_minimum_spacing(
    length: float, mesh_size: float
) -> tuple[int, float]:
    """Mirror the structured mesh's full cells plus optional remainder cell."""
    n_full = int(math.floor(length / mesh_size + 1.0e-12))
    remainder = length - n_full * mesh_size
    if math.isclose(remainder, 0.0, rel_tol=0.0, abs_tol=1.0e-9):
        return max(1, n_full), min(length, mesh_size)
    return n_full + 1, min(mesh_size, remainder)


def _block_counts(
    dimensions: tuple[float, float], mesh_size: float
) -> tuple[int, int, float]:
    nx, dx_min = _axis_count_and_minimum_spacing(dimensions[0], mesh_size)
    ny, dy_min = _axis_count_and_minimum_spacing(dimensions[1], mesh_size)
    nodes = (nx + 1) * (ny + 1)
    elements = ELEMENTS_PER_CELL * nx * ny
    return nodes, elements, min(dx_min, dy_min)


def estimate_case_size(config: PMMACaseConfig) -> dict[str, Any]:
    """Return conservative uncompressed sizes; HDF5 compression is not assumed."""
    mesh_size = config.numerics.mesh_size
    moving_nodes, moving_elements, moving_min_spacing = _block_counts(
        config.moving.dimensions, mesh_size
    )
    stationary_nodes, stationary_elements, stationary_min_spacing = _block_counts(
        config.stationary.dimensions, mesh_size
    )
    minimum_cell_size = min(moving_min_spacing, stationary_min_spacing)
    nodes = moving_nodes + stationary_nodes
    elements = moving_elements + stationary_elements
    bulk_frames = (
        config.output.bulk_normal_frames + config.output.bulk_shear_frames
    )
    interface_frames = (
        config.output.interface_normal_frames
        + config.output.interface_shear_frames
    )
    fault_cells, _ = _axis_count_and_minimum_spacing(
        config.moving.dimensions[1], mesh_size
    )
    fault_nodes = fault_cells + 1

    # Per bulk frame: displacement, optional velocity, stress, and optional strain.
    node_vector_count = 2 if config.output.store_bulk_velocity else 1
    element_tensor_count = 2 if config.output.store_bulk_strain else 1
    bytes_per_bulk_frame = 4 * (
        2 * node_vector_count * nodes + 4 * element_tensor_count * elements
    )
    # Seven dynamic interface arrays plus a 13-column history row.
    bytes_per_interface_frame = 4 * (7 * fault_nodes + 13)
    geometry_bytes = 4 * (
        2 * nodes + NODES_PER_ELEMENT * elements + 5 * fault_nodes
    )
    estimated_bytes = (
        bulk_frames * bytes_per_bulk_frame
        + interface_frames * bytes_per_interface_frame
        + geometry_bytes
    )
    tpv102_dof_ratio = 2 * nodes / TPV102_REFERENCE_DOFS
    equal_dof_mesh_estimate = mesh_size * tpv102_dof_ratio**0.5
    equal_dof_dump_estimate = estimated_bytes / max(tpv102_dof_ratio, 1.0e-12)
    material = config.material
    shear_modulus = material.young_modulus / (2.0 * (1.0 + material.poisson_ratio))
    lame_lambda = (
        material.young_modulus
        * material.poisson_ratio
        / ((1.0 + material.poisson_ratio) * (1.0 - 2.0 * material.poisson_ratio))
    )
    pressure_wave_speed = math.sqrt(
        (lame_lambda + 2.0 * shear_modulus) / material.density
    )
    bulk_dt_estimate = (
        config.numerics.cfl * minimum_cell_size / pressure_wave_speed
    )
    total_physical_time = (
        config.loading.normal_phase_time + config.loading.shear_phase_time
    )
    bulk_limited_steps_estimate = math.ceil(total_physical_time / bulk_dt_estimate)
    configured_steps_estimate = (
        None
        if config.numerics.time_step is None
        else math.ceil(total_physical_time / config.numerics.time_step)
    )
    return {
        "mesh_size_mm": mesh_size,
        "minimum_cell_size_mm": minimum_cell_size,
        "moving_nodes": moving_nodes,
        "stationary_nodes": stationary_nodes,
        "nodes_total": nodes,
        "degrees_of_freedom": 2 * nodes,
        "tpv102_reference_degrees_of_freedom": TPV102_REFERENCE_DOFS,
        "tpv102_dof_ratio": tpv102_dof_ratio,
        "mesh_for_tpv102_equal_dof_estimate_mm": equal_dof_mesh_estimate,
        "moving_elements": moving_elements,
        "stationary_elements": stationary_elements,
        "elements_total": elements,
        "fault_nodes": fault_nodes,
        "bulk_frames": bulk_frames,
        "interface_frames": interface_frames,
        "bytes_per_bulk_frame": bytes_per_bulk_frame,
        "bytes_per_interface_frame": bytes_per_interface_frame,
        "store_bulk_strain": config.output.store_bulk_strain,
        "store_bulk_velocity": config.output.store_bulk_velocity,
        "estimated_uncompressed_bytes": estimated_bytes,
        "estimated_uncompressed_gb": estimated_bytes / 1.0e9,
        "estimated_uncompressed_tb": estimated_bytes / 1.0e12,
        "equal_dof_dump_estimate_tb": equal_dof_dump_estimate / 1.0e12,
        "compression_assumption": "none (conservative upper estimate)",
        "pressure_wave_speed_mm_per_s": pressure_wave_speed,
        "bulk_cfl_dt_estimate_s": bulk_dt_estimate,
        "bulk_limited_steps_estimate": bulk_limited_steps_estimate,
        "configured_dt_s": config.numerics.time_step,
        "configured_steps_estimate": configured_steps_estimate,
        "contact_safety_factor": config.numerics.contact_safety_factor,
        "dt_estimate_note": (
            "The assembled contact limit is evaluated when the model is built; "
            "a configured time_step is accepted only when it does not exceed "
            "min(bulk_cfl_dt_estimate_s, dt_contact). Otherwise the actual dt is "
            "that minimum, and dt_contact scales linearly with "
            "contact_safety_factor."
        ),
    }
