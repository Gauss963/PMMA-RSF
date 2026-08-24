from __future__ import annotations

import json
import math
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tatva import Mesh, Operator
from tatva.element import Line2, Quad4, Tetrahedron4, Tri3
from tatva.friction import (
    project_regularized_rate_state_velocity,
    regularized_rate_state_strength,
    update_ageing_state,
    velocity_weakening_strengthening_coefficient,
)
from tatva.pmma.profiles import build_rate_state_profile
from tatva.pmma.model import (
    BlockSpec as LegacyBlockSpec,
    FrictionReference as LegacyFriction,
    LoadingReference as LegacySimulation,
    Material as LegacyMaterial,
    PMMAModelInput as LegacyCase,
)


FRICTION_LAWS = {
    "slip-weakening",
    "rate-state-vws",
    "rate-state-regularized",
}


class SimulationCheckpointed(RuntimeError):
    """Raised after a simulation state is safely checkpointed for later resume."""


def _write_simulation_checkpoint(
    path: Path,
    carry: tuple[jax.Array, ...],
    metadata: dict[str, Any],
) -> None:
    """Atomically persist the explicit integrator state outside the HDF5 dump."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = {
        f"carry_{index}": np.asarray(value)
        for index, value in enumerate(carry)
    }
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    with temporary.open("wb") as stream:
        np.savez(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_simulation_checkpoint(
    path: Path,
) -> tuple[dict[str, Any], tuple[jax.Array, ...]]:
    """Load one checkpoint without allowing object-array deserialization."""
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"]))
        carry = tuple(
            jnp.asarray(payload[f"carry_{index}"])
            for index in range(int(metadata["carry_count"]))
        )
    return metadata, carry


@dataclass(frozen=True)
class RunConfig:
    mesh_size: float
    simulation_time: float
    cfl: float
    dtype: str
    normal_penalty: float | None
    tangential_penalty: float | None
    contact_safety_factor: float = 0.25
    time_step_override: float | None = None
    operator_batch_size: int | None = None
    normal_loading_mode: str = "stress"
    normal_displacement_override: float | None = None
    shear_loading_mode: str = "stress"
    shear_loading_stiffness: float | None = None
    mu_k_override: float | None = None
    critical_slip_override: float | None = None
    loading_edge_nucleation_length: float = 0.0
    loading_edge_critical_slip: float | None = None
    shear_displacement_k_override: float | None = None
    shear_displacement_s_override: float | None = None
    output_prefix: str | None = None
    normal_phase_time: float | None = None
    shear_phase_time: float | None = None
    normal_ramp_time: float | None = None
    tau_k_start_fraction_override: float | None = 0.75
    tau_k_full_fraction_override: float | None = None
    shear_ramp_time: float | None = None
    shear_ramp_shape: str = "linear"
    normal_relaxation_time: float | None = None
    quasistatic_shear_fraction: float = 0.0
    quasistatic_shear_start_time: float = 0.0
    quasistatic_shear_ramp_time: float = 0.0
    quasistatic_damping_time: float | None = None
    stop_shear_loading_on_rupture: bool = False
    shear_loading_stop_slip: float | None = None
    shear_loading_stop_velocity: float | None = None
    shear_loading_stop_min_y: float | None = None
    shear_loading_stop_max_y: float | None = None
    relax_tangential_contact_during_normal: bool = False
    lock_shear_edge_during_normal: bool = False
    shear_scale: float = 1.0
    mu_s_start_fraction: float = 1.0
    mu_s_end_fraction: float = 1.0
    pw_length: float = 0.0
    pw_mu_s_ratio: float = 1.0
    pw_transition_length: float = 0.0
    leading_edge_guard_length: float = 0.0
    leading_edge_guard_mu_s_ratio: float = 1.0
    leading_edge_guard_transition_length: float = 0.0
    leading_edge_tangential_taper_length: float = 0.0
    leading_edge_tangential_plateau_length: float = 0.0
    leading_edge_tangential_taper_ratio: float = 1.0
    friction_law: str = "slip-weakening"
    rsf_reference_friction: float = 0.285
    rsf_direct_effect: float = 0.005
    rsf_state_effect: float = 0.0214
    rsf_reference_velocity: float = 1.0e-4
    rsf_reference_state: float = 3.3e-4
    rsf_characteristic_slip: float = 5.0e-4
    rsf_initial_state: float | None = None
    rsf_profile_spec: dict[str, Any] | None = None
    leading_edge_creep_length: float = 0.0
    leading_edge_creep_transition_length: float = 0.0
    leading_edge_creep_mu: float = 0.7
    leading_edge_creep_mu_k: float | None = None
    leading_edge_creep_relaxation_time: float = 0.0
    dimension: int = 2
    thickness: float = 0.0
    normal_stress_override: float | None = None
    shear_tau_k_override: float | None = None
    shear_tau_s_override: float | None = None


@dataclass(frozen=True)
class BlockModel:
    spec: LegacyBlockSpec
    mesh: Mesh
    operator: Operator
    boundary_nodes: dict[str, jax.Array]
    boundary_segments: dict[str, jax.Array]
    boundary_weights: dict[str, jax.Array]
    plot_elements: jax.Array
    plot_parent_elements: jax.Array

    @property
    def n_nodes(self) -> int:
        return int(self.mesh.coords.shape[0])


def _node_id(ix: int, iy: int, nx: int) -> int:
    return iy * (nx + 1) + ix


def _node_id_3d(ix: int, iy: int, iz: int, nx: int, ny: int) -> int:
    return iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix


def _axis_coordinates(start: float, length: float, mesh_size: float, dtype: jnp.dtype) -> jax.Array:
    if mesh_size <= 0.0:
        raise ValueError(f"mesh_size must be positive, got {mesh_size}")
    n_full = int(math.floor(length / mesh_size + 1e-12))
    coords = [start + i * mesh_size for i in range(n_full + 1)]
    end = start + length
    if not math.isclose(coords[-1], end, rel_tol=0.0, abs_tol=1e-9):
        coords.append(end)
    return jnp.asarray(coords, dtype=dtype)


def _structured_2d_boundary(
    spec: LegacyBlockSpec, nx: int, ny: int
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    def make_edge_nodes(which: str) -> jax.Array:
        if which == "back":
            return jnp.arange(0, (ny + 1) * (nx + 1), nx + 1, dtype=jnp.int32)
        if which == "front":
            return jnp.arange(nx, (ny + 1) * (nx + 1) + nx, nx + 1, dtype=jnp.int32)
        if which == "right":
            return jnp.arange(nx + 1, dtype=jnp.int32)
        if which == "left":
            return jnp.arange(ny * (nx + 1), (ny + 1) * (nx + 1), dtype=jnp.int32)
        raise ValueError(which)

    boundary_nodes = {
        f"{spec.name}-back": make_edge_nodes("back"),
        f"{spec.name}-front": make_edge_nodes("front"),
        f"{spec.name}-right": make_edge_nodes("right"),
        f"{spec.name}-left": make_edge_nodes("left"),
    }
    boundary_segments = {
        name: jnp.stack([nodes[:-1], nodes[1:]], axis=-1)
        for name, nodes in boundary_nodes.items()
    }
    return boundary_nodes, boundary_segments


def _structured_2d_grid(
    spec: LegacyBlockSpec, mesh_size: float, dtype: jnp.dtype
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int, int]:
    x0, y0 = spec.origin
    lx, ly = spec.dimensions
    x_vals = _axis_coordinates(x0, lx, mesh_size, dtype)
    y_vals = _axis_coordinates(y0, ly, mesh_size, dtype)
    nx = int(x_vals.shape[0] - 1)
    ny = int(y_vals.shape[0] - 1)
    X, Y = jnp.meshgrid(x_vals, y_vals, indexing="xy")
    coords = jnp.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)

    ix, iy = jnp.meshgrid(
        jnp.arange(nx, dtype=jnp.int32),
        jnp.arange(ny, dtype=jnp.int32),
        indexing="xy",
    )
    n00 = (iy * (nx + 1) + ix).reshape(-1)
    n10 = n00 + 1
    n01 = n00 + (nx + 1)
    n11 = n01 + 1
    return n00, n10, n01, n11, coords, nx, ny


def create_structured_tri_block(
    spec: LegacyBlockSpec, mesh_size: float, dtype: jnp.dtype
) -> tuple[Mesh, dict[str, jax.Array], dict[str, jax.Array]]:
    n00, n10, n01, n11, coords, nx, ny = _structured_2d_grid(spec, mesh_size, dtype)
    tri1 = jnp.stack([n00, n10, n11], axis=-1)
    tri2 = jnp.stack([n00, n11, n01], axis=-1)
    elements = jnp.concatenate([tri1, tri2], axis=0)
    boundary_nodes, boundary_segments = _structured_2d_boundary(spec, nx, ny)
    return Mesh(coords=coords, elements=elements), boundary_nodes, boundary_segments


def create_structured_quad_block(
    spec: LegacyBlockSpec, mesh_size: float, dtype: jnp.dtype
) -> tuple[Mesh, dict[str, jax.Array], dict[str, jax.Array]]:
    n00, n10, n01, n11, coords, nx, ny = _structured_2d_grid(spec, mesh_size, dtype)
    elements = jnp.stack([n00, n10, n11, n01], axis=-1)
    boundary_nodes, boundary_segments = _structured_2d_boundary(spec, nx, ny)
    return Mesh(coords=coords, elements=elements), boundary_nodes, boundary_segments


def create_structured_tet_block(
    spec: LegacyBlockSpec,
    mesh_size: float,
    thickness: float,
    dtype: jnp.dtype,
) -> tuple[Mesh, dict[str, jax.Array], dict[str, jax.Array], jax.Array, jax.Array]:
    if thickness <= 0.0:
        raise ValueError(f"3D thickness must be positive, got {thickness}")

    x0, y0 = spec.origin
    lx, ly = spec.dimensions
    x_vals = np.asarray(_axis_coordinates(x0, lx, mesh_size, dtype), dtype=np.float64)
    y_vals = np.asarray(_axis_coordinates(y0, ly, mesh_size, dtype), dtype=np.float64)
    z_vals = np.asarray(_axis_coordinates(0.0, thickness, mesh_size, dtype), dtype=np.float64)
    nx = len(x_vals) - 1
    ny = len(y_vals) - 1
    nz = len(z_vals) - 1

    coords = np.empty(((nx + 1) * (ny + 1) * (nz + 1), 3), dtype=np.float32)
    for iz, z in enumerate(z_vals):
        for iy, y in enumerate(y_vals):
            for ix, x in enumerate(x_vals):
                coords[_node_id_3d(ix, iy, iz, nx, ny)] = (x, y, z)

    elements: list[list[int]] = []
    boundary_faces: dict[str, list[list[int]]] = {
        f"{spec.name}-back": [],
        f"{spec.name}-front": [],
        f"{spec.name}-right": [],
        f"{spec.name}-left": [],
        f"{spec.name}-bottom": [],
        f"{spec.name}-top": [],
    }
    boundary_face_parents: dict[str, list[int]] = {name: [] for name in boundary_faces}

    def add_face(name: str, tri: list[int], parent_idx: int) -> None:
        boundary_faces[name].append(tri)
        boundary_face_parents[name].append(parent_idx)

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                n000 = _node_id_3d(ix, iy, iz, nx, ny)
                n100 = _node_id_3d(ix + 1, iy, iz, nx, ny)
                n010 = _node_id_3d(ix, iy + 1, iz, nx, ny)
                n110 = _node_id_3d(ix + 1, iy + 1, iz, nx, ny)
                n001 = _node_id_3d(ix, iy, iz + 1, nx, ny)
                n101 = _node_id_3d(ix + 1, iy, iz + 1, nx, ny)
                n011 = _node_id_3d(ix, iy + 1, iz + 1, nx, ny)
                n111 = _node_id_3d(ix + 1, iy + 1, iz + 1, nx, ny)

                cube_tets = [
                    [n000, n100, n110, n111],
                    [n000, n110, n010, n111],
                    [n000, n010, n011, n111],
                    [n000, n011, n001, n111],
                    [n000, n001, n101, n111],
                    [n000, n101, n100, n111],
                ]
                base_parent = len(elements)
                elements.extend(cube_tets)

                if ix == 0:
                    add_face(f"{spec.name}-back", [n000, n010, n011], base_parent + 2)
                    add_face(f"{spec.name}-back", [n000, n011, n001], base_parent + 3)
                if ix == nx - 1:
                    add_face(f"{spec.name}-front", [n100, n111, n110], base_parent + 0)
                    add_face(f"{spec.name}-front", [n100, n101, n111], base_parent + 5)
                if iy == 0:
                    add_face(f"{spec.name}-right", [n000, n101, n100], base_parent + 5)
                    add_face(f"{spec.name}-right", [n000, n001, n101], base_parent + 4)
                if iy == ny - 1:
                    add_face(f"{spec.name}-left", [n010, n110, n111], base_parent + 1)
                    add_face(f"{spec.name}-left", [n010, n111, n011], base_parent + 2)
                if iz == 0:
                    add_face(f"{spec.name}-bottom", [n000, n100, n110], base_parent + 0)
                    add_face(f"{spec.name}-bottom", [n000, n110, n010], base_parent + 1)
                if iz == nz - 1:
                    add_face(f"{spec.name}-top", [n001, n011, n111], base_parent + 3)
                    add_face(f"{spec.name}-top", [n001, n111, n101], base_parent + 4)

    boundary_segments = {
        name: jnp.asarray(tris, dtype=jnp.int32) for name, tris in boundary_faces.items()
    }
    boundary_nodes = {
        name: jnp.unique(tris.reshape(-1)).astype(jnp.int32)
        for name, tris in boundary_segments.items()
    }
    plot_elements = boundary_segments[f"{spec.name}-bottom"]
    plot_parent_elements = jnp.asarray(
        boundary_face_parents[f"{spec.name}-bottom"], dtype=jnp.int32
    )
    return (
        Mesh(coords=jnp.asarray(coords, dtype=dtype), elements=jnp.asarray(elements, dtype=jnp.int32)),
        boundary_nodes,
        boundary_segments,
        plot_elements,
        plot_parent_elements,
    )


def make_boundary_operator(mesh: Mesh, segments: jax.Array) -> Operator:
    n_nodes = int(segments.shape[1])
    if n_nodes == 2:
        return Operator(mesh._replace(elements=segments), Line2())
    if n_nodes == 3:
        return Operator(mesh._replace(elements=segments), Tri3())
    raise ValueError(f"Unsupported boundary element with {n_nodes} nodes")


def boundary_weights(mesh: Mesh, segments: jax.Array, dtype: jnp.dtype) -> jax.Array:
    if mesh.coords.shape[1] == 3 and segments.shape[1] == 3:
        tri_pts = mesh.coords[segments]
        area = 0.5 * jnp.linalg.norm(
            jnp.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0]),
            axis=1,
        )
        weights = jnp.zeros(mesh.coords.shape[0], dtype=dtype)
        nodal = area / 3.0
        for local_idx in range(3):
            weights = weights.at[segments[:, local_idx]].add(nodal)
        return weights
    boundary_op = make_boundary_operator(mesh, segments)
    zeros = jnp.zeros(mesh.coords.shape[0], dtype=dtype)
    return jax.jacrev(lambda q: boundary_op.integrate(q))(zeros)


def build_block_model(
    spec: LegacyBlockSpec,
    mesh_size: float,
    dtype: jnp.dtype,
    *,
    dimension: int,
    thickness: float,
    operator_batch_size: int | None = None,
) -> BlockModel:
    if dimension == 2:
        mesh, boundary_nodes, boundary_segments = create_structured_quad_block(
            spec, mesh_size, dtype
        )
        batch_size = (
            None
            if operator_batch_size is None
            else min(operator_batch_size, int(mesh.elements.shape[0]))
        )
        operator = Operator(mesh, Quad4(), batch_size=batch_size)
        plot_elements = mesh.elements
        plot_parent_elements = jnp.arange(mesh.elements.shape[0], dtype=jnp.int32)
    elif dimension == 3:
        mesh, boundary_nodes, boundary_segments, plot_elements, plot_parent_elements = (
            create_structured_tet_block(spec, mesh_size, thickness, dtype)
        )
        batch_size = (
            None
            if operator_batch_size is None
            else min(operator_batch_size, int(mesh.elements.shape[0]))
        )
        operator = Operator(mesh, Tetrahedron4(), batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported dimension {dimension}")
    weights = {
        name: boundary_weights(mesh, segments, dtype)
        for name, segments in boundary_segments.items()
    }
    return BlockModel(
        spec=spec,
        mesh=mesh,
        operator=operator,
        boundary_nodes=boundary_nodes,
        boundary_segments=boundary_segments,
        boundary_weights=weights,
        plot_elements=plot_elements,
        plot_parent_elements=plot_parent_elements,
    )


def lumped_mass(operator: Operator, density: float, dtype: jnp.dtype) -> jax.Array:
    # Assemble the row-sum mass explicitly. Differentiating one global integral gives
    # the same weights, but its reverse pass becomes unreliable for very large meshes.
    shape_at_quads = jax.vmap(operator.element.shape_function)(
        operator.element.quad_points
    )
    integration_weights = operator.get_integration_weights()
    local_weights = jnp.einsum("eq,qi->ei", integration_weights, shape_at_quads)
    weights = jnp.zeros(operator.mesh.coords.shape[0], dtype=dtype).at[
        operator.mesh.elements.reshape(-1)
    ].add(local_weights.reshape(-1))
    nonpositive_nodes = int(jnp.count_nonzero(weights <= 0.0))
    if nonpositive_nodes:
        nonpositive_quads = int(jnp.count_nonzero(integration_weights <= 0.0))
        nonpositive_local = int(jnp.count_nonzero(local_weights <= 0.0))
        first_element = int(
            jnp.flatnonzero(integration_weights <= 0.0, size=1)[0]
            // integration_weights.shape[1]
        )
        first_element_nodes = operator.mesh.elements[first_element]
        first_element_coords = np.asarray(
            operator.mesh.coords[first_element_nodes]
        ).tolist()
        direct_det = float(
            operator.element.get_jacobian(
                operator.element.quad_points[0],
                operator.mesh.coords[first_element_nodes],
            )[1]
        )
        first_node = int(jnp.flatnonzero(weights <= 0.0, size=1)[0])
        first_coords = np.asarray(operator.mesh.coords[first_node]).tolist()
        raise ValueError(
            "Lumped-mass assembly produced non-positive nodal weights: "
            f"nodes={nonpositive_nodes}, quadrature_weights={nonpositive_quads}, "
            f"local_weights={nonpositive_local}, first_element={first_element}, "
            f"first_element_nodes={np.asarray(first_element_nodes).tolist()}, "
            f"first_element_coords={first_element_coords}, direct_det={direct_det}, "
            f"first_node={first_node}, "
            f"first_coords={first_coords}, min_quadrature_weight="
            f"{float(jnp.min(integration_weights))}, min_local_weight="
            f"{float(jnp.min(local_weights))}."
        )
    return weights * jnp.asarray(density, dtype=dtype)


def quadrature_weighted_element_average(
    values: jax.Array, integration_weights: jax.Array
) -> jax.Array:
    """Collapse quadrature-point values to one volume-weighted value per element."""
    if values.ndim < 3 or integration_weights.ndim != 2:
        raise ValueError(
            "Expected values shaped (elements, quadrature, ...) and weights "
            "shaped (elements, quadrature)."
        )
    if values.shape[:2] != integration_weights.shape:
        raise ValueError(
            "Quadrature values and integration weights must share their first "
            f"two dimensions, got {values.shape[:2]} and {integration_weights.shape}."
        )
    weighted = jnp.einsum("eq...,eq->e...", values, integration_weights)
    element_measure = jnp.sum(integration_weights, axis=1)
    denominator = element_measure.reshape(
        (element_measure.shape[0],) + (1,) * (values.ndim - 2)
    )
    return weighted / denominator


def make_global_dof_indices(
    nodes: jax.Array | np.ndarray, offset: int, component: int, n_components: int
) -> jax.Array:
    return offset + n_components * nodes + component


def make_dirichlet_dofs(
    stationary: BlockModel, moving_offset: int, *, dimension: int
) -> jax.Array:
    front_nodes = stationary.boundary_nodes["stationary-block-front"]
    top_nodes = stationary.boundary_nodes["stationary-block-left"]
    dofs = [
        make_global_dof_indices(front_nodes, moving_offset, 0, dimension),
        make_global_dof_indices(top_nodes, moving_offset, 1, dimension),
    ]
    if dimension == 3:
        bottom_nodes = stationary.boundary_nodes["stationary-block-bottom"]
        dofs.append(make_global_dof_indices(bottom_nodes, moving_offset, 2, dimension))
    return jnp.unique(jnp.concatenate(dofs).astype(jnp.int32))


def match_interface_nodes(
    master: BlockModel, slave: BlockModel, master_surface: str, slave_surface: str
) -> tuple[jax.Array, jax.Array, jax.Array]:
    master_nodes = master.boundary_nodes[master_surface]
    slave_nodes = slave.boundary_nodes[slave_surface]
    master_coords = np.asarray(master.mesh.coords[master_nodes])
    slave_coords = np.asarray(slave.mesh.coords[slave_nodes])
    tangential_dims = list(range(1, master.mesh.coords.shape[1]))
    master_tangent = master_coords[:, tangential_dims]
    slave_tangent = slave_coords[:, tangential_dims]
    master_rows = np.round(master_tangent, decimals=6)
    slave_lookup = {
        tuple(row.tolist()): int(slave_nodes[idx])
        for idx, row in enumerate(np.round(slave_tangent, decimals=6))
    }
    matched_master: list[int] = []
    matched_slave: list[int] = []
    for idx, row in enumerate(master_rows):
        key = tuple(row.tolist())
        if key in slave_lookup:
            matched_master.append(int(master_nodes[idx]))
            matched_slave.append(slave_lookup[key])
    if not matched_master:
        raise ValueError("No overlapping interface nodes were found.")
    return (
        jnp.asarray(matched_master, dtype=jnp.int32),
        jnp.asarray(matched_slave, dtype=jnp.int32),
    )


def select_interface_plot_nodes(
    block: BlockModel,
    interface_nodes: jax.Array,
) -> jax.Array:
    coords = np.asarray(block.mesh.coords[np.asarray(interface_nodes)])
    if coords.shape[1] == 2:
        order = np.argsort(coords[:, 1])
        return jnp.asarray(np.asarray(interface_nodes)[order], dtype=jnp.int32)

    z_vals = coords[:, 2]
    z_mid = 0.5 * (float(z_vals.min()) + float(z_vals.max()))
    z_pick = float(z_vals[np.argmin(np.abs(z_vals - z_mid))])
    line_mask = np.isclose(z_vals, z_pick, atol=1e-6)
    selected = np.asarray(interface_nodes)[line_mask]
    selected_coords = np.asarray(block.mesh.coords[selected])
    order = np.argsort(selected_coords[:, 1])
    return jnp.asarray(selected[order], dtype=jnp.int32)


def build_mu_s_profile(
    block: BlockModel,
    interface_nodes: jax.Array,
    friction: LegacyFriction,
    mu_s_start_fraction: float,
    mu_s_end_fraction: float,
    pw_length: float,
    pw_mu_s_ratio: float,
    pw_transition_length: float,
    leading_edge_guard_length: float,
    leading_edge_guard_mu_s_ratio: float,
    leading_edge_guard_transition_length: float,
    dtype: jnp.dtype,
) -> jax.Array:
    start_fraction = float(mu_s_start_fraction)
    end_fraction = float(mu_s_end_fraction)
    if start_fraction < 0.0:
        raise ValueError(f"mu_s_start_fraction must be non-negative, got {start_fraction}")
    if end_fraction < 0.0:
        raise ValueError(f"mu_s_end_fraction must be non-negative, got {end_fraction}")

    coords = np.asarray(block.mesh.coords[np.asarray(interface_nodes)])
    y_coords = coords[:, 1]
    y_min = float(y_coords.min())
    y_max = float(y_coords.max())
    interface_length = y_max - y_min
    pw_length = float(pw_length)
    pw_mu_s_ratio = float(pw_mu_s_ratio)
    pw_transition_length = float(pw_transition_length)
    if pw_length < 0.0:
        raise ValueError(f"pw_length must be non-negative, got {pw_length}")
    if pw_transition_length < 0.0:
        raise ValueError(
            "pw_transition_length must be non-negative, "
            f"got {pw_transition_length}"
        )
    if pw_mu_s_ratio < 0.0:
        raise ValueError(f"pw_mu_s_ratio must be non-negative, got {pw_mu_s_ratio}")
    if pw_length + pw_transition_length > interface_length + 1e-12:
        raise ValueError(
            "pw_length + pw_transition_length cannot exceed the interface length. "
            f"Got {pw_length + pw_transition_length:.6g} for an interface of "
            f"{interface_length:.6g}."
        )
    if math.isclose(y_min, y_max, rel_tol=0.0, abs_tol=1e-12):
        progress = np.zeros_like(y_coords, dtype=np.float64)
    else:
        progress = (y_coords - y_min) / (y_max - y_min)
    mu_s_profile = friction.mu_s * (
        start_fraction + (end_fraction - start_fraction) * progress
    )
    if pw_length > 0.0 and not math.isclose(
        pw_mu_s_ratio,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        pw_start = y_max - pw_length
        if pw_transition_length > 0.0:
            transition_start = pw_start - pw_transition_length
            transition_progress = np.clip(
                (y_coords - transition_start) / pw_transition_length,
                0.0,
                1.0,
            )
            blend = 0.5 * (1.0 - np.cos(np.pi * transition_progress))
            pw_factor = 1.0 + (pw_mu_s_ratio - 1.0) * blend
        else:
            pw_factor = np.where(y_coords >= pw_start, pw_mu_s_ratio, 1.0)
        mu_s_profile *= pw_factor

    guard_length = float(leading_edge_guard_length)
    guard_ratio = float(leading_edge_guard_mu_s_ratio)
    guard_transition = float(leading_edge_guard_transition_length)
    if guard_length < 0.0:
        raise ValueError(
            f"leading_edge_guard_length must be non-negative, got {guard_length}"
        )
    if guard_transition < 0.0:
        raise ValueError(
            "leading_edge_guard_transition_length must be non-negative, "
            f"got {guard_transition}"
        )
    if guard_ratio < 1.0:
        raise ValueError(
            "leading_edge_guard_mu_s_ratio must be at least 1.0, "
            f"got {guard_ratio}"
        )
    if guard_length + guard_transition > interface_length + 1e-12:
        raise ValueError(
            "leading-edge guard length + transition length cannot exceed the "
            f"interface length. Got {guard_length + guard_transition:.6g} "
            f"for an interface of {interface_length:.6g}."
        )
    if guard_length > 0.0 and not math.isclose(
        guard_ratio,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        guard_start = y_max - guard_length
        if guard_transition > 0.0:
            transition_start = guard_start - guard_transition
            transition_progress = np.clip(
                (y_coords - transition_start) / guard_transition,
                0.0,
                1.0,
            )
            blend = 0.5 * (1.0 - np.cos(np.pi * transition_progress))
            guard_factor = 1.0 + (guard_ratio - 1.0) * blend
        else:
            guard_factor = np.where(y_coords >= guard_start, guard_ratio, 1.0)
        mu_s_profile *= guard_factor
    if float(mu_s_profile.min()) + 1e-12 < friction.mu_k:
        raise ValueError(
            "The static friction profile makes the static friction fall below mu_k. "
            f"Minimum mu_s would be {float(mu_s_profile.min()):.6g} while mu_k={friction.mu_k:.6g}."
        )
    return jnp.asarray(mu_s_profile, dtype=dtype)


def build_critical_slip_profile(
    moving: BlockModel,
    master_nodes: jax.Array,
    critical_slip: float,
    loading_edge_nucleation_length: float,
    loading_edge_critical_slip: float | None,
    dtype: jnp.dtype,
) -> jax.Array:
    if loading_edge_nucleation_length < 0.0:
        raise ValueError("loading_edge_nucleation_length must be non-negative.")
    edge_critical_slip = (
        critical_slip
        if loading_edge_critical_slip is None
        else float(loading_edge_critical_slip)
    )
    if edge_critical_slip <= 0.0:
        raise ValueError("loading_edge_critical_slip must be positive.")
    if loading_edge_nucleation_length == 0.0:
        if loading_edge_critical_slip is not None and not math.isclose(
            edge_critical_slip, critical_slip
        ):
            raise ValueError(
                "loading_edge_nucleation_length must be positive when "
                "loading_edge_critical_slip differs from the fault critical slip."
            )
        return jnp.full(master_nodes.shape, critical_slip, dtype=dtype)

    contact_y = np.asarray(moving.mesh.coords[master_nodes, 1], dtype=float)
    distance_from_loading_edge = contact_y - float(contact_y.min())
    phase = np.clip(
        distance_from_loading_edge / loading_edge_nucleation_length,
        0.0,
        1.0,
    )
    edge_weight = 0.5 * (1.0 + np.cos(np.pi * phase))
    profile = critical_slip + (edge_critical_slip - critical_slip) * edge_weight
    return jnp.asarray(profile, dtype=dtype)


def build_friction_profiles(
    block: BlockModel,
    interface_nodes: jax.Array,
    mu_s_profile: jax.Array,
    friction: LegacyFriction,
    creep_length: float,
    creep_transition_length: float,
    creep_mu: float,
    creep_mu_k: float | None,
    dtype: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Blend the leading edge into a viscoplastic Coulomb creep zone."""
    creep_length = float(creep_length)
    transition_length = float(creep_transition_length)
    creep_mu = float(creep_mu)
    resolved_creep_mu_k = creep_mu if creep_mu_k is None else float(creep_mu_k)
    if creep_length < 0.0:
        raise ValueError("leading_edge_creep_length must be non-negative.")
    if transition_length < 0.0:
        raise ValueError("leading_edge_creep_transition_length must be non-negative.")
    if creep_mu <= 0.0:
        raise ValueError("leading_edge_creep_mu must be positive.")
    if resolved_creep_mu_k <= 0.0:
        raise ValueError("leading_edge_creep_mu_k must be positive.")

    y_coords = np.asarray(block.mesh.coords[np.asarray(interface_nodes), 1], dtype=float)
    interface_length = float(np.max(y_coords) - np.min(y_coords))
    if creep_length + transition_length > interface_length + 1e-12:
        raise ValueError(
            "leading-edge creep length + transition length cannot exceed the "
            f"interface length. Got {creep_length + transition_length:.6g} "
            f"for an interface of {interface_length:.6g}."
        )

    mu_s = np.asarray(mu_s_profile, dtype=float)
    mu_k = np.full_like(mu_s, friction.mu_k)
    creep_weight = np.zeros_like(mu_s)
    if creep_length > 0.0:
        creep_start = float(np.max(y_coords)) - creep_length
        if transition_length > 0.0:
            transition_start = creep_start - transition_length
            progress = np.clip(
                (y_coords - transition_start) / transition_length,
                0.0,
                1.0,
            )
            blend = 0.5 * (1.0 - np.cos(np.pi * progress))
        else:
            blend = np.where(y_coords >= creep_start, 1.0, 0.0)
        creep_weight = blend
        mu_s = mu_s + (creep_mu - mu_s) * blend
        mu_k = mu_k + (resolved_creep_mu_k - mu_k) * blend

    if np.any(mu_s + 1e-12 < mu_k):
        raise ValueError("The local static-friction profile cannot fall below local kinetic friction.")
    return (
        jnp.asarray(mu_s, dtype=dtype),
        jnp.asarray(mu_k, dtype=dtype),
        jnp.asarray(creep_weight, dtype=dtype),
    )


def mesh_min_edge_length(mesh: Mesh) -> float:
    pts = mesh.coords[mesh.elements]
    n_nodes_per_element = int(mesh.elements.shape[1])
    edge_vectors = []
    for i in range(n_nodes_per_element):
        for j in range(i + 1, n_nodes_per_element):
            edge_vectors.append(pts[:, j] - pts[:, i])
    edges = jnp.concatenate(edge_vectors, axis=0)
    return float(jnp.min(jnp.linalg.norm(edges, axis=1)))


def build_tangential_penalty_profile(
    block: BlockModel,
    interface_nodes: jax.Array,
    base_penalty: float,
    taper_length: float,
    plateau_length: float,
    taper_ratio: float,
    dtype: Any,
) -> jax.Array:
    """Smoothly soften tangential contact stiffness near the leading edge."""
    if taper_length < 0.0:
        raise ValueError("leading_edge_tangential_taper_length must be non-negative.")
    if plateau_length < 0.0:
        raise ValueError("leading_edge_tangential_plateau_length must be non-negative.")
    if not 0.0 < taper_ratio <= 1.0:
        raise ValueError(
            "leading_edge_tangential_taper_ratio must be greater than 0 and at most 1."
        )

    y_coords = np.asarray(block.mesh.coords[np.asarray(interface_nodes), 1], dtype=float)
    factors = np.ones_like(y_coords)
    if not math.isclose(taper_ratio, 1.0):
        y_max = float(np.max(y_coords))
        plateau_start = y_max - plateau_length
        if taper_length > 0.0:
            transition_start = plateau_start - taper_length
            progress = np.clip(
                (y_coords - transition_start) / taper_length,
                0.0,
                1.0,
            )
            blend = 0.5 * (1.0 - np.cos(np.pi * progress))
            factors += (taper_ratio - 1.0) * blend
        elif plateau_length > 0.0:
            factors = np.where(y_coords >= plateau_start, taper_ratio, 1.0)
    return jnp.asarray(base_penalty * factors, dtype=dtype)


def estimate_normal_displacement(
    case: LegacyCase,
    *,
    normal_stress: float,
    dimension: int,
) -> float:
    moving_material = case.materials["moving-block"]
    stationary_material = case.materials["stationary-block"]
    moving_length = float(case.moving.dimensions[0])
    stationary_length = float(case.stationary.dimensions[0])

    if dimension == 2:
        moving_compliance = (1.0 - moving_material.nu**2) / moving_material.E
        stationary_compliance = (1.0 - stationary_material.nu**2) / stationary_material.E
    elif dimension == 3:
        moving_compliance = 1.0 / moving_material.E
        stationary_compliance = 1.0 / stationary_material.E
    else:
        raise ValueError(f"Unsupported dimension {dimension}")

    return float(
        normal_stress
        * (
            moving_length * moving_compliance
            + stationary_length * stationary_compliance
        )
    )


def build_ramp_progress(
    num_steps: int,
    *,
    shape: str,
    dtype: type[np.floating[Any]],
) -> np.ndarray:
    """Return a normalized loading ramp with exact zero/one endpoints."""
    x = np.linspace(0.0, 1.0, num=num_steps, endpoint=True, dtype=dtype)
    if shape == "linear":
        progress = x
    elif shape == "smoothstep":
        progress = x * x * (3.0 - 2.0 * x)
    elif shape == "half-cosine":
        progress = 0.5 * (1.0 - np.cos(np.pi * x))
    else:
        raise ValueError(
            "shear_ramp_shape must be 'linear', 'smoothstep', or 'half-cosine', "
            f"got {shape!r}."
        )
    return np.asarray(progress, dtype=dtype)


def _shear_loading_stop_candidates(
    cumulative_slip: jax.Array,
    slip_rate: jax.Array,
    stop_mask: jax.Array,
    critical_slip_profile: jax.Array,
    stop_slip: jax.Array,
    uses_critical_profile: bool,
    stop_velocity: jax.Array | None,
) -> jax.Array:
    """Return stations that satisfy the configured dynamic rupture trigger."""
    slip_limit = critical_slip_profile if uses_critical_profile else stop_slip
    candidates = stop_mask & (cumulative_slip >= slip_limit)
    if stop_velocity is not None:
        candidates = candidates & (jnp.abs(slip_rate) >= stop_velocity)
    return candidates


def build_case_model(case: LegacyCase, config: RunConfig) -> dict[str, Any]:
    dtype = jnp.float32 if config.dtype == "float32" else jnp.float64
    dimension = int(config.dimension)
    if dimension not in (2, 3):
        raise ValueError(f"Unsupported dimension {dimension}")
    if config.operator_batch_size is not None and config.operator_batch_size <= 0:
        raise ValueError("operator_batch_size must be positive when specified.")
    if config.contact_safety_factor <= 0.0:
        raise ValueError("contact_safety_factor must be positive.")
    if config.time_step_override is not None and config.time_step_override <= 0.0:
        raise ValueError("time_step_override must be positive when specified.")

    moving = build_block_model(
        case.moving,
        config.mesh_size,
        dtype,
        dimension=dimension,
        thickness=config.thickness,
        operator_batch_size=config.operator_batch_size,
    )
    stationary = build_block_model(
        case.stationary,
        config.mesh_size,
        dtype,
        dimension=dimension,
        thickness=config.thickness,
        operator_batch_size=config.operator_batch_size,
    )

    moving_material = case.materials["moving-block"]
    stationary_material = case.materials["stationary-block"]
    friction = LegacyFriction(
        mu_s=case.friction.mu_s,
        mu_k=(
            case.friction.mu_k
            if config.mu_k_override is None
            else float(config.mu_k_override)
        ),
        d_c=case.friction.d_c,
    )
    if friction.mu_k <= 0.0 or friction.mu_k > friction.mu_s:
        raise ValueError("mu_k_override must be positive and no greater than mu_s.")
    normal_stress = (
        case.simulation.normal_stress
        if config.normal_stress_override is None
        else float(config.normal_stress_override)
    )

    moving_mass = lumped_mass(moving.operator, moving_material.rho, dtype)
    stationary_mass = lumped_mass(stationary.operator, stationary_material.rho, dtype)

    moving_n = moving.n_nodes
    stationary_n = stationary.n_nodes
    moving_offset = dimension * moving_n
    total_dofs = dimension * (moving_n + stationary_n)

    fixed_dofs = make_dirichlet_dofs(stationary, moving_offset, dimension=dimension)
    moving_normal_edge_dofs = make_global_dof_indices(
        moving.boundary_nodes["moving-block-back"],
        0,
        0,
        dimension,
    ).astype(jnp.int32)
    moving_shear_edge_dofs = make_global_dof_indices(
        moving.boundary_nodes["moving-block-right"],
        0,
        1,
        dimension,
    ).astype(jnp.int32)
    moving_shear_loading_dofs = moving_shear_edge_dofs

    mass_flat = jnp.concatenate(
        [
            jnp.repeat(moving_mass, dimension),
            jnp.repeat(stationary_mass, dimension),
        ]
    )

    force_normal = jnp.zeros(total_dofs, dtype=dtype).at[
        make_global_dof_indices(
            jnp.arange(moving_n, dtype=jnp.int32),
            0,
            0,
            dimension,
        )
    ].add(
        moving.boundary_weights["moving-block-back"]
        * jnp.asarray(normal_stress, dtype=dtype)
    )

    force_shear_unit = jnp.zeros(total_dofs, dtype=dtype).at[
        make_global_dof_indices(
            jnp.arange(moving_n, dtype=jnp.int32),
            0,
            1,
            dimension,
        )
    ].add(moving.boundary_weights["moving-block-right"])

    master_nodes, slave_nodes = match_interface_nodes(
        moving,
        stationary,
        case.simulation.master_surface,
        case.simulation.slave_surface,
    )
    mu_s_profile = build_mu_s_profile(
        moving,
        master_nodes,
        friction,
        config.mu_s_start_fraction,
        config.mu_s_end_fraction,
        config.pw_length,
        config.pw_mu_s_ratio,
        config.pw_transition_length,
        config.leading_edge_guard_length,
        config.leading_edge_guard_mu_s_ratio,
        config.leading_edge_guard_transition_length,
        dtype,
    )
    mu_s_profile, mu_k_profile, creep_weight_profile = build_friction_profiles(
        moving,
        master_nodes,
        mu_s_profile,
        friction,
        config.leading_edge_creep_length,
        config.leading_edge_creep_transition_length,
        config.leading_edge_creep_mu,
        config.leading_edge_creep_mu_k,
        dtype,
    )
    if config.leading_edge_creep_relaxation_time < 0.0:
        raise ValueError("leading_edge_creep_relaxation_time must be non-negative.")
    friction_law = str(config.friction_law).strip().lower().replace("_", "-")
    if friction_law not in FRICTION_LAWS:
        raise ValueError(
            "friction_law must be 'slip-weakening', 'rate-state-vws', or "
            "'rate-state-regularized', "
            f"got {config.friction_law!r}."
        )
    rsf_parameters: dict[str, Any] = {
        "reference_friction": float(config.rsf_reference_friction),
        "direct_effect": float(config.rsf_direct_effect),
        "state_effect": float(config.rsf_state_effect),
        "reference_velocity": float(config.rsf_reference_velocity),
        "reference_state": float(config.rsf_reference_state),
        "characteristic_slip": float(config.rsf_characteristic_slip),
        "initial_state": float(
            config.rsf_reference_state
            if config.rsf_initial_state is None
            else config.rsf_initial_state
        ),
    }
    if friction_law in {"rate-state-vws", "rate-state-regularized"}:
        positive_rsf_parameters = (
            "reference_friction",
            "reference_velocity",
            "reference_state",
            "characteristic_slip",
            "initial_state",
        )
        for name in positive_rsf_parameters:
            if rsf_parameters[name] <= 0.0:
                raise ValueError(f"rsf_{name} must be positive.")
        if rsf_parameters["direct_effect"] < 0.0:
            raise ValueError("rsf_direct_effect must be non-negative.")
        if rsf_parameters["state_effect"] < 0.0:
            raise ValueError("rsf_state_effect must be non-negative.")
        incompatible_settings = []
        if config.mu_k_override is not None:
            incompatible_settings.append("mu_k_override")
        if config.critical_slip_override is not None:
            incompatible_settings.append("critical_slip_override")
        if config.loading_edge_nucleation_length > 0.0:
            incompatible_settings.append("loading_edge_nucleation_length")
        if config.loading_edge_critical_slip is not None:
            incompatible_settings.append("loading_edge_critical_slip")
        if config.mu_s_start_fraction != 1.0 or config.mu_s_end_fraction != 1.0:
            incompatible_settings.append("mu_s spatial fractions")
        if config.pw_length > 0.0 or config.pw_transition_length > 0.0:
            incompatible_settings.append("pw friction profile")
        if (
            config.leading_edge_guard_length > 0.0
            or config.leading_edge_guard_transition_length > 0.0
        ):
            incompatible_settings.append("leading-edge friction guard")
        if (
            config.leading_edge_tangential_taper_length > 0.0
            or config.leading_edge_tangential_plateau_length > 0.0
        ):
            incompatible_settings.append("leading-edge tangential taper")
        if (
            config.leading_edge_creep_length > 0.0
            or config.leading_edge_creep_transition_length > 0.0
            or config.leading_edge_creep_relaxation_time > 0.0
        ):
            incompatible_settings.append("legacy leading-edge creep")
        if incompatible_settings:
            raise ValueError(
                "rate-state friction cannot be combined with slip-weakening or "
                "terminal-interface modifications: "
                + ", ".join(incompatible_settings)
            )
        if (
            config.stop_shear_loading_on_rupture
            and config.shear_loading_stop_slip is None
        ):
            raise ValueError(
                "rate-state friction requires an explicit shear_loading_stop_slip "
                "when automatic loading stop is enabled."
            )
        if config.rsf_profile_spec is not None:
            interface_y = np.asarray(moving.mesh.coords[np.asarray(master_nodes), 1])
            profile = build_rate_state_profile(interface_y, config.rsf_profile_spec)
            rsf_parameters = {
                name: jnp.asarray(profile[name], dtype=dtype)
                for name in (
                    "reference_friction",
                    "direct_effect",
                    "state_effect",
                    "reference_velocity",
                    "reference_state",
                    "characteristic_slip",
                    "initial_state",
                )
            }
            for name in positive_rsf_parameters:
                if np.any(np.asarray(rsf_parameters[name]) <= 0.0):
                    raise ValueError(f"RSF profile field {name} must be positive.")
            if np.any(np.asarray(rsf_parameters["direct_effect"]) < 0.0):
                raise ValueError("RSF profile field direct_effect must be non-negative.")
            if np.any(np.asarray(rsf_parameters["state_effect"]) < 0.0):
                raise ValueError("RSF profile field state_effect must be non-negative.")
    interface_plot_master_nodes = select_interface_plot_nodes(moving, master_nodes)
    interface_plot_slave_nodes = select_interface_plot_nodes(stationary, slave_nodes)
    interface_weights = moving.boundary_weights[case.simulation.master_surface][master_nodes]

    penalty_n = (
        config.normal_penalty
        if config.normal_penalty is not None
        else 10.0 * moving_material.E / config.mesh_size
    )
    tangential_penalty = (
        config.tangential_penalty
        if config.tangential_penalty is not None
        else penalty_n * 0.1
    )
    penalty_t = build_tangential_penalty_profile(
        moving,
        master_nodes,
        tangential_penalty,
        config.leading_edge_tangential_taper_length,
        config.leading_edge_tangential_plateau_length,
        config.leading_edge_tangential_taper_ratio,
        dtype,
    )

    hmin = min(mesh_min_edge_length(moving.mesh), mesh_min_edge_length(stationary.mesh))
    cp = max(moving_material.cp, stationary_material.cp)
    dt_bulk = config.cfl * hmin / cp
    min_mass = float(jnp.min(mass_flat[jnp.setdiff1d(jnp.arange(total_dofs), fixed_dofs)]))
    max_interface_weight = float(jnp.max(interface_weights))
    max_contact_penalty = max(float(penalty_n), float(jnp.max(penalty_t)))
    dt_contact = config.contact_safety_factor * math.sqrt(
        min_mass / max_contact_penalty / max_interface_weight
    )
    stable_dt = min(dt_bulk, dt_contact)
    stability_limiter = "bulk" if dt_bulk <= dt_contact else "contact"
    if (
        config.time_step_override is not None
        and config.time_step_override > stable_dt * (1.0 + 1.0e-12)
    ):
        raise ValueError(
            "Requested time_step_override exceeds the assembled stability limit: "
            f"requested={config.time_step_override}, stable_limit={stable_dt}, "
            f"dt_bulk={dt_bulk}, dt_contact={dt_contact}."
        )
    dt = (
        stable_dt
        if config.time_step_override is None
        else float(config.time_step_override)
    )
    dt_limiter = (
        stability_limiter if config.time_step_override is None else "configured"
    )
    if not math.isfinite(dt) or dt <= 0.0:
        nonpositive_mass_count = int(jnp.count_nonzero(mass_flat <= 0.0))
        raise ValueError(
            "Stable time-step estimate is not positive: "
            f"dt={dt}, dt_bulk={dt_bulk}, dt_contact={dt_contact}, "
            f"hmin={hmin}, cp={cp}, min_mass={min_mass}, "
            f"nonpositive_mass_count={nonpositive_mass_count}, "
            f"max_contact_penalty={max_contact_penalty}, "
            f"max_interface_weight={max_interface_weight}."
        )

    pressure_time = (
        config.normal_phase_time
        if config.normal_phase_time is not None
        else config.simulation_time
    )
    shear_time = (
        config.shear_phase_time
        if config.shear_phase_time is not None
        else config.simulation_time
    )

    pressure_steps = max(1, int(math.ceil(pressure_time / dt)))
    shear_steps = max(1, int(math.ceil(shear_time / dt)))
    normal_ramp_time = (
        0.0
        if config.normal_ramp_time is None
        else min(max(float(config.normal_ramp_time), 0.0), pressure_time)
    )
    normal_ramp_steps = (
        min(pressure_steps, int(math.ceil(normal_ramp_time / dt)))
        if normal_ramp_time > 0.0
        else 0
    )
    normal_loading_mode = str(config.normal_loading_mode).strip().lower()
    if normal_loading_mode not in {"stress", "displacement"}:
        raise ValueError(
            f"normal_loading_mode must be 'stress' or 'displacement', got {config.normal_loading_mode!r}"
        )
    shear_loading_mode = str(config.shear_loading_mode).strip().lower()
    if shear_loading_mode not in {"stress", "displacement", "spring-displacement"}:
        raise ValueError(
            "shear_loading_mode must be 'stress', 'displacement', or "
            f"'spring-displacement', got {config.shear_loading_mode!r}"
        )
    normal_displacement_estimate = estimate_normal_displacement(
        case,
        normal_stress=normal_stress,
        dimension=dimension,
    )
    normal_displacement = (
        normal_displacement_estimate
        if config.normal_displacement_override is None
        else float(config.normal_displacement_override)
    )
    tau_k_start_fraction = (
        case.simulation.tau_k_start_fraction
        if config.tau_k_start_fraction_override is None
        else min(max(float(config.tau_k_start_fraction_override), 0.0), 1.0)
    )
    tau_k_full_fraction_raw = (
        case.simulation.tau_k_start_fraction
        if config.tau_k_full_fraction_override is None
        else min(max(float(config.tau_k_full_fraction_override), 0.0), 1.0)
    )
    tau_k_full_fraction = max(tau_k_start_fraction, tau_k_full_fraction_raw)
    tau_k_start_step = min(
        pressure_steps - 1,
        max(0, int(math.floor(tau_k_start_fraction * pressure_steps))),
    )
    tau_k_full_step = min(
        pressure_steps - 1,
        max(0, int(math.floor(tau_k_full_fraction * pressure_steps))),
    )
    shear_ratio = case.moving.dimensions[1] / case.stationary.dimensions[0]
    tau_scale = max(float(config.shear_scale), 0.0)
    tau_k = (
        float(config.shear_tau_k_override)
        if config.shear_tau_k_override is not None
        else tau_scale * shear_ratio * friction.mu_k * normal_stress
    )
    tau_s = (
        float(config.shear_tau_s_override)
        if config.shear_tau_s_override is not None
        else tau_scale * shear_ratio * friction.mu_s * normal_stress
    )
    if (
        shear_loading_mode in {"displacement", "spring-displacement"}
        and config.shear_displacement_k_override is None
    ):
        raise ValueError(
            "shear_displacement_k_override must be provided for displacement-controlled loading."
        )
    shear_loading_stiffness = (
        None
        if config.shear_loading_stiffness is None
        else float(config.shear_loading_stiffness)
    )
    if shear_loading_mode == "spring-displacement" and (
        shear_loading_stiffness is None or shear_loading_stiffness <= 0.0
    ):
        raise ValueError(
            "spring-displacement loading requires a positive shear_loading_stiffness."
        )
    shear_displacement_k = (
        0.0
        if config.shear_displacement_k_override is None
        else float(config.shear_displacement_k_override)
    )
    shear_displacement_s = (
        shear_displacement_k
        if config.shear_displacement_s_override is None
        else float(config.shear_displacement_s_override)
    )
    shear_ramp_time = (
        case.simulation.rise_fraction * shear_time
        if config.shear_ramp_time is None
        else min(max(float(config.shear_ramp_time), 0.0), shear_time)
    )
    shear_ramp_steps = (
        min(shear_steps, int(math.ceil(shear_ramp_time / dt)))
        if shear_ramp_time > 0.0
        else 0
    )
    shear_ramp_shape = str(config.shear_ramp_shape).strip().lower()
    if shear_ramp_shape not in {"linear", "smoothstep", "half-cosine"}:
        raise ValueError(
            "shear_ramp_shape must be 'linear', 'smoothstep', or 'half-cosine', "
            f"got {config.shear_ramp_shape!r}."
        )
    quasistatic_shear_fraction = float(config.quasistatic_shear_fraction)
    quasistatic_shear_start_time = float(config.quasistatic_shear_start_time)
    quasistatic_shear_ramp_time = float(config.quasistatic_shear_ramp_time)
    quasistatic_damping_time = (
        None
        if config.quasistatic_damping_time is None
        else float(config.quasistatic_damping_time)
    )
    normal_relaxation_time = (
        None
        if config.normal_relaxation_time is None
        else float(config.normal_relaxation_time)
    )
    if (
        normal_relaxation_time is not None
        and quasistatic_damping_time is not None
        and not math.isclose(
            normal_relaxation_time,
            quasistatic_damping_time,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        )
    ):
        raise ValueError(
            "normal_relaxation_time and the legacy quasistatic_damping_time "
            "must match when both are provided."
        )
    if normal_relaxation_time is None:
        normal_relaxation_time = quasistatic_damping_time
    if normal_relaxation_time is not None and normal_relaxation_time <= 0.0:
        raise ValueError("normal_relaxation_time must be positive when enabled.")
    if not 0.0 <= quasistatic_shear_fraction < 1.0:
        raise ValueError("quasistatic_shear_fraction must be in [0, 1).")
    if quasistatic_shear_fraction > 0.0:
        if shear_loading_mode != "displacement":
            raise ValueError(
                "quasistatic shear preloading requires displacement-controlled loading."
            )
        if quasistatic_shear_start_time < 0.0:
            raise ValueError("quasistatic_shear_start_time cannot be negative.")
        if quasistatic_shear_ramp_time <= 0.0:
            raise ValueError(
                "quasistatic_shear_ramp_time must be positive when preloading is enabled."
            )
        if (
            quasistatic_shear_start_time + quasistatic_shear_ramp_time
            > pressure_time
        ):
            raise ValueError(
                "quasistatic shear ramp must finish within normal_phase_time."
            )
        if normal_relaxation_time is None:
            raise ValueError(
                "normal_relaxation_time must be provided when quasistatic shear "
                "preloading is enabled."
            )
    critical_slip = (
        float(friction.d_c)
        if config.critical_slip_override is None
        else float(config.critical_slip_override)
    )
    if critical_slip <= 0.0:
        raise ValueError("critical_slip_override must be positive.")
    critical_slip_profile = build_critical_slip_profile(
        moving,
        master_nodes,
        critical_slip,
        config.loading_edge_nucleation_length,
        config.loading_edge_critical_slip,
        dtype,
    )
    shear_loading_stop_uses_critical_profile = (
        config.shear_loading_stop_slip is None
    )
    shear_loading_stop_slip = (
        critical_slip
        if shear_loading_stop_uses_critical_profile
        else float(config.shear_loading_stop_slip)
    )
    if shear_loading_stop_slip <= 0.0:
        raise ValueError("shear_loading_stop_slip must be positive.")
    shear_loading_stop_velocity = (
        None
        if config.shear_loading_stop_velocity is None
        else float(config.shear_loading_stop_velocity)
    )
    if (
        shear_loading_stop_velocity is not None
        and shear_loading_stop_velocity <= 0.0
    ):
        raise ValueError("shear_loading_stop_velocity must be positive.")
    stop_min_y = (
        -math.inf
        if config.shear_loading_stop_min_y is None
        else float(config.shear_loading_stop_min_y)
    )
    stop_max_y = (
        math.inf
        if config.shear_loading_stop_max_y is None
        else float(config.shear_loading_stop_max_y)
    )
    if stop_min_y > stop_max_y:
        raise ValueError("shear_loading_stop_min_y must not exceed stop_max_y.")
    contact_y = moving.mesh.coords[master_nodes, 1]
    shear_loading_stop_mask = (contact_y >= stop_min_y) & (
        contact_y <= stop_max_y
    )
    if not bool(np.asarray(jnp.any(shear_loading_stop_mask))):
        raise ValueError("The shear-loading stop y-range contains no contact nodes.")
    if config.stop_shear_loading_on_rupture and shear_loading_mode not in {
        "displacement",
        "spring-displacement",
    }:
        raise ValueError(
            "stop_shear_loading_on_rupture requires displacement-controlled loading."
        )

    scalar_dtype = np.float32 if dtype == jnp.float32 else np.float64
    shear_ramp_progress = (
        build_ramp_progress(
            shear_ramp_steps,
            shape=shear_ramp_shape,
            dtype=scalar_dtype,
        )
        if shear_ramp_steps > 0
        else np.zeros(0, dtype=scalar_dtype)
    )
    pressure_schedule = np.zeros(pressure_steps, dtype=scalar_dtype)
    pressure_schedule[tau_k_full_step:] = tau_k
    if tau_k_start_step >= tau_k_full_step:
        pressure_schedule[tau_k_start_step:] = tau_k
    else:
        pressure_ramp_steps = tau_k_full_step - tau_k_start_step + 1
        pressure_schedule[tau_k_start_step : tau_k_full_step + 1] = np.linspace(
            tau_k / pressure_ramp_steps,
            tau_k,
            num=pressure_ramp_steps,
            endpoint=True,
            dtype=scalar_dtype,
        )
    shear_schedule = np.full(shear_steps, tau_s, dtype=scalar_dtype)
    if shear_ramp_steps <= 0:
        shear_schedule[:] = tau_s
    else:
        shear_schedule[:shear_ramp_steps] = (
            tau_k + (tau_s - tau_k) * shear_ramp_progress
        )
    shear_displacement_pressure = np.zeros(pressure_steps, dtype=scalar_dtype)
    if tau_k_start_step >= tau_k_full_step:
        shear_displacement_pressure[tau_k_start_step:] = shear_displacement_k
    else:
        shear_displacement_pressure[tau_k_full_step:] = shear_displacement_k
        pressure_ramp_steps = tau_k_full_step - tau_k_start_step + 1
        shear_displacement_pressure[tau_k_start_step : tau_k_full_step + 1] = np.linspace(
            shear_displacement_k / pressure_ramp_steps,
            shear_displacement_k,
            num=pressure_ramp_steps,
            endpoint=True,
            dtype=scalar_dtype,
        )
    shear_displacement_shear = np.full(shear_steps, shear_displacement_s, dtype=scalar_dtype)
    quasistatic_shear_target = shear_displacement_k + quasistatic_shear_fraction * (
        shear_displacement_s - shear_displacement_k
    )
    if quasistatic_shear_fraction > 0.0:
        preload_start_step = min(
            pressure_steps - 1,
            max(0, int(math.floor(quasistatic_shear_start_time / dt))),
        )
        preload_ramp_steps = max(
            2,
            int(math.ceil(quasistatic_shear_ramp_time / dt)) + 1,
        )
        preload_stop_step = min(
            pressure_steps,
            preload_start_step + preload_ramp_steps,
        )
        preload_progress = build_ramp_progress(
            preload_stop_step - preload_start_step,
            shape=shear_ramp_shape,
            dtype=scalar_dtype,
        )
        shear_displacement_pressure[preload_start_step:preload_stop_step] = (
            shear_displacement_k
            + (quasistatic_shear_target - shear_displacement_k) * preload_progress
        )
        shear_displacement_pressure[preload_stop_step:] = quasistatic_shear_target
    if shear_ramp_steps <= 0:
        shear_displacement_shear[:] = shear_displacement_s
    else:
        shear_displacement_shear[:shear_ramp_steps] = (
            quasistatic_shear_target
            + (shear_displacement_s - quasistatic_shear_target)
            * shear_ramp_progress
        )
    if shear_loading_mode in {"displacement", "spring-displacement"}:
        pressure_schedule[:] = 0.0
        shear_schedule[:] = 0.0
    else:
        shear_displacement_pressure[:] = 0.0
        shear_displacement_shear[:] = 0.0

    normal_schedule_pressure = np.ones(pressure_steps, dtype=scalar_dtype)
    if normal_ramp_steps > 0:
        normal_schedule_pressure[:normal_ramp_steps] = (
            np.arange(1, normal_ramp_steps + 1, dtype=scalar_dtype) / normal_ramp_steps
        )
    normal_schedule_shear = np.ones(shear_steps, dtype=scalar_dtype)
    normal_displacement_pressure = np.zeros(pressure_steps, dtype=scalar_dtype)
    normal_displacement_shear = np.zeros(shear_steps, dtype=scalar_dtype)
    if normal_loading_mode == "displacement":
        normal_schedule_pressure[:] = 0.0
        normal_schedule_shear[:] = 0.0
        if normal_ramp_steps > 0:
            normal_displacement_pressure[:normal_ramp_steps] = (
                np.arange(1, normal_ramp_steps + 1, dtype=scalar_dtype)
                / normal_ramp_steps
                * normal_displacement
            )
            normal_displacement_pressure[normal_ramp_steps:] = normal_displacement
        else:
            normal_displacement_pressure[:] = normal_displacement
        normal_displacement_shear[:] = normal_displacement

    normal_velocity_pressure = np.zeros(pressure_steps, dtype=scalar_dtype)
    prev_disp = scalar_dtype(0.0)
    for idx in range(pressure_steps):
        disp_now = normal_displacement_pressure[idx]
        normal_velocity_pressure[idx] = (disp_now - prev_disp) / dt
        prev_disp = disp_now
    normal_velocity_shear = np.zeros(shear_steps, dtype=scalar_dtype)
    prev_disp = normal_displacement_pressure[-1] if pressure_steps > 0 else scalar_dtype(0.0)
    for idx in range(shear_steps):
        disp_now = normal_displacement_shear[idx]
        normal_velocity_shear[idx] = (disp_now - prev_disp) / dt
        prev_disp = disp_now
    shear_velocity_pressure = np.zeros(pressure_steps, dtype=scalar_dtype)
    prev_disp = scalar_dtype(0.0)
    for idx in range(pressure_steps):
        disp_now = shear_displacement_pressure[idx]
        shear_velocity_pressure[idx] = (disp_now - prev_disp) / dt
        prev_disp = disp_now
    shear_velocity_shear = np.zeros(shear_steps, dtype=scalar_dtype)
    prev_disp = shear_displacement_pressure[-1] if pressure_steps > 0 else scalar_dtype(0.0)
    for idx in range(shear_steps):
        disp_now = shear_displacement_shear[idx]
        shear_velocity_shear[idx] = (disp_now - prev_disp) / dt
        prev_disp = disp_now

    return {
        "dtype": dtype,
        "moving": moving,
        "stationary": stationary,
        "moving_material": moving_material,
        "stationary_material": stationary_material,
        "friction": friction,
        "fixed_dofs": fixed_dofs,
        "moving_normal_edge_dofs": moving_normal_edge_dofs,
        "moving_shear_edge_dofs": moving_shear_edge_dofs,
        "moving_shear_loading_dofs": moving_shear_loading_dofs,
        "shear_force_boundary": "moving-block-right",
        "shear_displacement_boundary": "moving-block-right",
        "force_normal": force_normal,
        "force_shear_unit": force_shear_unit,
        "mass_flat": mass_flat,
        "master_nodes": master_nodes,
        "slave_nodes": slave_nodes,
        "interface_plot_master_nodes": interface_plot_master_nodes,
        "interface_plot_slave_nodes": interface_plot_slave_nodes,
        "interface_weights": interface_weights,
        "mu_s_profile": mu_s_profile,
        "mu_k_profile": mu_k_profile,
        "critical_slip_profile": critical_slip_profile,
        "creep_weight_profile": creep_weight_profile,
        "friction_law": friction_law,
        "rsf_parameters": rsf_parameters,
        "rsf_profile_spec": config.rsf_profile_spec,
        "penalty_n": jnp.asarray(penalty_n, dtype=dtype),
        "tangential_penalty": float(tangential_penalty),
        "penalty_t": jnp.asarray(penalty_t, dtype=dtype),
        "cfl": float(config.cfl),
        "contact_safety_factor": float(config.contact_safety_factor),
        "time_step_override": config.time_step_override,
        "dt": float(dt),
        "dt_stable_limit": float(stable_dt),
        "dt_bulk": float(dt_bulk),
        "dt_contact": float(dt_contact),
        "dt_stability_limiter": stability_limiter,
        "dt_limiter": dt_limiter,
        "pressure_schedule": jnp.asarray(pressure_schedule, dtype=dtype),
        "shear_schedule": jnp.asarray(shear_schedule, dtype=dtype),
        "normal_schedule_pressure": jnp.asarray(normal_schedule_pressure, dtype=dtype),
        "normal_schedule_shear": jnp.asarray(normal_schedule_shear, dtype=dtype),
        "normal_displacement_pressure": jnp.asarray(normal_displacement_pressure, dtype=dtype),
        "normal_displacement_shear": jnp.asarray(normal_displacement_shear, dtype=dtype),
        "normal_velocity_pressure": jnp.asarray(normal_velocity_pressure, dtype=dtype),
        "normal_velocity_shear": jnp.asarray(normal_velocity_shear, dtype=dtype),
        "shear_displacement_pressure": jnp.asarray(shear_displacement_pressure, dtype=dtype),
        "shear_displacement_shear": jnp.asarray(shear_displacement_shear, dtype=dtype),
        "shear_velocity_pressure": jnp.asarray(shear_velocity_pressure, dtype=dtype),
        "shear_velocity_shear": jnp.asarray(shear_velocity_shear, dtype=dtype),
        "tau_k": float(tau_k),
        "tau_s": float(tau_s),
        "tau_k_start_fraction": float(tau_k_start_fraction),
        "tau_k_full_fraction": float(tau_k_full_fraction),
        "shear_scale": tau_scale,
        "normal_loading_mode": normal_loading_mode,
        "shear_loading_mode": shear_loading_mode,
        "shear_loading_stiffness": shear_loading_stiffness,
        "critical_slip": critical_slip,
        "mu_k": friction.mu_k,
        "loading_edge_nucleation_length": float(
            config.loading_edge_nucleation_length
        ),
        "loading_edge_critical_slip": (
            critical_slip
            if config.loading_edge_critical_slip is None
            else float(config.loading_edge_critical_slip)
        ),
        "mu_s_start_fraction": float(config.mu_s_start_fraction),
        "mu_s_end_fraction": float(config.mu_s_end_fraction),
        "pw_length": float(config.pw_length),
        "pw_mu_s_ratio": float(config.pw_mu_s_ratio),
        "pw_transition_length": float(config.pw_transition_length),
        "leading_edge_guard_length": float(config.leading_edge_guard_length),
        "leading_edge_guard_mu_s_ratio": float(
            config.leading_edge_guard_mu_s_ratio
        ),
        "leading_edge_guard_transition_length": float(
            config.leading_edge_guard_transition_length
        ),
        "leading_edge_tangential_taper_length": float(
            config.leading_edge_tangential_taper_length
        ),
        "leading_edge_tangential_plateau_length": float(
            config.leading_edge_tangential_plateau_length
        ),
        "leading_edge_tangential_taper_ratio": float(
            config.leading_edge_tangential_taper_ratio
        ),
        "rsf_reference_friction": float(
            np.mean(np.asarray(rsf_parameters["reference_friction"]))
        ),
        "rsf_direct_effect": float(
            np.mean(np.asarray(rsf_parameters["direct_effect"]))
        ),
        "rsf_state_effect": float(
            np.mean(np.asarray(rsf_parameters["state_effect"]))
        ),
        "rsf_reference_velocity": float(
            np.mean(np.asarray(rsf_parameters["reference_velocity"]))
        ),
        "rsf_reference_state": float(
            np.mean(np.asarray(rsf_parameters["reference_state"]))
        ),
        "rsf_characteristic_slip": float(
            np.mean(np.asarray(rsf_parameters["characteristic_slip"]))
        ),
        "rsf_initial_state": float(
            np.mean(np.asarray(rsf_parameters["initial_state"]))
        ),
        "leading_edge_creep_length": float(config.leading_edge_creep_length),
        "leading_edge_creep_transition_length": float(
            config.leading_edge_creep_transition_length
        ),
        "leading_edge_creep_mu": float(config.leading_edge_creep_mu),
        "leading_edge_creep_mu_k": float(
            config.leading_edge_creep_mu
            if config.leading_edge_creep_mu_k is None
            else config.leading_edge_creep_mu_k
        ),
        "leading_edge_creep_relaxation_time": float(
            config.leading_edge_creep_relaxation_time
        ),
        "normal_stress": float(normal_stress),
        "normal_displacement": float(normal_displacement),
        "normal_displacement_estimate": float(normal_displacement_estimate),
        "shear_displacement_k": float(shear_displacement_k),
        "shear_displacement_s": float(shear_displacement_s),
        "quasistatic_shear_fraction": quasistatic_shear_fraction,
        "quasistatic_shear_target": float(quasistatic_shear_target),
        "quasistatic_shear_start_time": quasistatic_shear_start_time,
        "quasistatic_shear_ramp_time": quasistatic_shear_ramp_time,
        "normal_relaxation_time": normal_relaxation_time,
        # Retain the old key for readers of existing summary payloads.
        "quasistatic_damping_time": normal_relaxation_time,
        "pressure_time": float(pressure_time),
        "shear_time": float(shear_time),
        "shear_ramp_time": float(shear_ramp_time),
        "shear_ramp_shape": shear_ramp_shape,
        "stop_shear_loading_on_rupture": bool(
            config.stop_shear_loading_on_rupture
        ),
        "shear_loading_stop_slip": shear_loading_stop_slip,
        "shear_loading_stop_velocity": shear_loading_stop_velocity,
        "shear_loading_stop_uses_critical_profile": (
            shear_loading_stop_uses_critical_profile
        ),
        "shear_loading_stop_min_y": config.shear_loading_stop_min_y,
        "shear_loading_stop_max_y": config.shear_loading_stop_max_y,
        "shear_loading_stop_mask": shear_loading_stop_mask,
        "relax_tangential_contact_during_normal": bool(
            config.relax_tangential_contact_during_normal
        ),
        "normal_ramp_time": float(normal_ramp_time),
        "normal_ramp_steps": normal_ramp_steps,
        "shear_ramp_steps": shear_ramp_steps,
        "pressure_steps": pressure_steps,
        "shear_steps": shear_steps,
        "moving_offset": moving_offset,
        "total_dofs": total_dofs,
        "dimension": dimension,
        "thickness": float(config.thickness),
    }


def run_simulation(case: LegacyCase, config: RunConfig) -> dict[str, Any]:
    model = build_case_model(case, config)

    dtype = model["dtype"]
    moving = model["moving"]
    stationary = model["stationary"]
    moving_material = model["moving_material"]
    stationary_material = model["stationary_material"]
    base_fixed_dofs = model["fixed_dofs"]
    moving_normal_edge_dofs = model["moving_normal_edge_dofs"]
    moving_shear_edge_dofs = model["moving_shear_edge_dofs"]
    moving_shear_loading_dofs = model["moving_shear_loading_dofs"]
    force_normal = model["force_normal"]
    force_shear_unit = model["force_shear_unit"]
    mass_flat = model["mass_flat"]
    master_nodes = model["master_nodes"]
    slave_nodes = model["slave_nodes"]
    interface_weights = model["interface_weights"]
    mu_s_profile = model["mu_s_profile"]
    mu_k_profile = model["mu_k_profile"]
    critical_slip_profile = model["critical_slip_profile"]
    creep_weight_profile = model["creep_weight_profile"]
    creep_relaxation_time = jnp.asarray(
        model["leading_edge_creep_relaxation_time"], dtype=dtype
    )
    if model["friction_law"] == "rate-state-regularized":
        raise ValueError(
            "rate-state-regularized requires run_simulation_dumped(), which "
            "applies the validated implicit TPV velocity projection."
        )
    use_rate_state = model["friction_law"] == "rate-state-vws"
    rsf_parameters = {
        name: jnp.broadcast_to(jnp.asarray(value, dtype=dtype), master_nodes.shape)
        for name, value in model["rsf_parameters"].items()
    }
    penalty_n = model["penalty_n"]
    penalty_t = model["penalty_t"]
    dt = model["dt"]
    moving_offset = model["moving_offset"]
    total_dofs = model["total_dofs"]
    dimension = int(model["dimension"])
    interface_plot_master_nodes = model["interface_plot_master_nodes"]
    normal_loading_mode = str(model["normal_loading_mode"])
    use_normal_displacement = normal_loading_mode == "displacement"
    shear_loading_mode = str(model["shear_loading_mode"])
    use_shear_displacement = shear_loading_mode == "displacement"
    use_shear_spring = shear_loading_mode == "spring-displacement"
    shear_loading_stiffness = jnp.asarray(
        0.0
        if model["shear_loading_stiffness"] is None
        else model["shear_loading_stiffness"],
        dtype=dtype,
    )
    stop_shear_loading = bool(model["stop_shear_loading_on_rupture"])
    shear_loading_stop_slip = jnp.asarray(
        model["shear_loading_stop_slip"], dtype=dtype
    )
    shear_loading_stop_velocity = (
        None
        if model["shear_loading_stop_velocity"] is None
        else jnp.asarray(model["shear_loading_stop_velocity"], dtype=dtype)
    )
    shear_loading_stop_uses_critical_profile = bool(
        model["shear_loading_stop_uses_critical_profile"]
    )
    shear_loading_stop_mask = model["shear_loading_stop_mask"]
    relax_tangential_contact_during_normal = bool(
        model["relax_tangential_contact_during_normal"]
    )
    quasistatic_preloading = model["quasistatic_shear_fraction"] > 0.0
    normal_relaxation = model["normal_relaxation_time"] is not None
    normal_relaxation_factor = jnp.asarray(
        (
            1.0
            if not normal_relaxation
            else math.exp(-dt / model["normal_relaxation_time"])
        ),
        dtype=dtype,
    )

    n_moving = moving.n_nodes
    n_stationary = stationary.n_nodes
    friction = model["friction"]

    if config.lock_shear_edge_during_normal:
        normal_phase_zero_dofs = jnp.unique(
            jnp.concatenate([base_fixed_dofs, moving_shear_edge_dofs])
        )
    else:
        normal_phase_zero_dofs = base_fixed_dofs
    shear_phase_zero_dofs = base_fixed_dofs
    prescribed_normal_dofs = (
        moving_normal_edge_dofs
        if use_normal_displacement
        else jnp.zeros((0,), dtype=jnp.int32)
    )
    prescribed_shear_dofs = (
        moving_shear_loading_dofs
        if use_shear_displacement
        else jnp.zeros((0,), dtype=jnp.int32)
    )
    prescribed_dofs = jnp.concatenate([prescribed_normal_dofs, prescribed_shear_dofs])

    def apply_constraints(
        vec: jax.Array,
        zero_dofs: jax.Array,
        prescribed_dofs: jax.Array,
        prescribed_values: jax.Array,
    ) -> jax.Array:
        constrained = vec.at[zero_dofs].set(0.0)
        return constrained.at[prescribed_dofs].set(prescribed_values)

    def zero_constrained_dofs(
        vec: jax.Array,
        zero_dofs: jax.Array,
        prescribed_dofs: jax.Array,
    ) -> jax.Array:
        constrained = vec.at[zero_dofs].set(0.0)
        return constrained.at[prescribed_dofs].set(0.0)

    def split_u(u_flat: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (
            u_flat[: dimension * n_moving].reshape(n_moving, dimension),
            u_flat[dimension * n_moving :].reshape(n_stationary, dimension),
        )

    def elastic_energy_total(u_flat: jax.Array) -> jax.Array:
        u_moving, u_stationary = split_u(u_flat)
        eps_moving = moving.operator.grad(u_moving)
        eps_stationary = stationary.operator.grad(u_stationary)

        def strain_energy_density(grad_u: jax.Array, mat: LegacyMaterial) -> jax.Array:
            eps = 0.5 * (grad_u + jnp.swapaxes(grad_u, -1, -2))
            return mat.mu * jnp.einsum("...ij,...ij->...", eps, eps) + 0.5 * mat.lmbda * jnp.trace(
                eps, axis1=-2, axis2=-1
            ) ** 2

        return moving.operator.integrate(
            strain_energy_density(eps_moving, moving_material)
        ) + stationary.operator.integrate(
            strain_energy_density(eps_stationary, stationary_material)
        )

    elastic_energy_and_force = jax.jit(jax.value_and_grad(elastic_energy_total))

    total_interface_length = jnp.sum(interface_weights)
    moving_iface_x = dimension * master_nodes
    moving_iface_y = dimension * master_nodes + 1
    stationary_iface_x = moving_offset + dimension * slave_nodes
    stationary_iface_y = moving_offset + dimension * slave_nodes + 1

    def contact_response(
        u_flat: jax.Array,
        v_flat: jax.Array,
        plastic_slip: jax.Array,
        cum_slip: jax.Array,
        rsf_state: jax.Array,
        tangential_friction_active: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        u_moving, u_stationary = split_u(u_flat)
        v_moving, v_stationary = split_u(v_flat)
        rel_normal = u_moving[master_nodes, 0] - u_stationary[slave_nodes, 0]
        penetration = jnp.maximum(rel_normal, 0.0)
        in_contact = penetration > 0.0

        rel_tangent = u_moving[master_nodes, 1] - u_stationary[slave_nodes, 1]
        trial_tau = penalty_t * (rel_tangent - plastic_slip)
        normal_traction = penalty_n * penetration
        slip_weakening_mu = jnp.maximum(
            mu_k_profile,
            mu_s_profile
            - (mu_s_profile - mu_k_profile)
            * jnp.minimum(cum_slip / critical_slip_profile, 1.0),
        )
        friction_velocity = jnp.abs(
            v_moving[master_nodes, 1] - v_stationary[slave_nodes, 1]
        )
        if use_rate_state:
            evolved_rsf_state = update_ageing_state(
                rsf_state,
                friction_velocity,
                dt,
                rsf_parameters["characteristic_slip"],
            )
            new_rsf_state = jnp.where(
                in_contact, evolved_rsf_state, rsf_state
            )
            mu_eff = velocity_weakening_strengthening_coefficient(
                friction_velocity,
                new_rsf_state,
                reference_friction=rsf_parameters["reference_friction"],
                direct_effect=rsf_parameters["direct_effect"],
                state_effect=rsf_parameters["state_effect"],
                reference_velocity=rsf_parameters["reference_velocity"],
                reference_state=rsf_parameters["reference_state"],
            )
        else:
            new_rsf_state = rsf_state
            mu_eff = slip_weakening_mu
        friction_strength = jnp.where(
            in_contact, mu_eff * normal_traction, 0.0
        )
        yield_tau = friction_strength
        sliding = in_contact & (jnp.abs(trial_tau) > yield_tau)
        plastic_correction = (
            jnp.sign(trial_tau)
            * jnp.maximum(jnp.abs(trial_tau) - yield_tau, 0.0)
            / penalty_t
        )
        viscous_fraction = dt / (dt + creep_relaxation_time)
        correction_fraction = 1.0 - creep_weight_profile * (1.0 - viscous_fraction)
        plastic_increment = jnp.where(
            sliding, correction_fraction * plastic_correction, 0.0
        )
        new_plastic = plastic_slip + plastic_increment
        tau = jnp.where(
            in_contact,
            trial_tau - penalty_t * plastic_increment,
            0.0,
        )
        new_cum = cum_slip + jnp.abs(plastic_increment)
        new_slip_rate = jnp.abs(plastic_increment) / dt
        tau = jnp.where(tangential_friction_active, tau, 0.0)
        new_plastic = jnp.where(
            tangential_friction_active, new_plastic, rel_tangent
        )
        new_cum = jnp.where(tangential_friction_active, new_cum, cum_slip)
        new_slip_rate = jnp.where(
            tangential_friction_active, new_slip_rate, 0.0
        )

        forces = jnp.zeros(total_dofs, dtype=dtype)
        forces = forces.at[moving_iface_x].add(interface_weights * normal_traction)
        forces = forces.at[stationary_iface_x].add(-interface_weights * normal_traction)
        forces = forces.at[moving_iface_y].add(interface_weights * tau)
        forces = forces.at[stationary_iface_y].add(-interface_weights * tau)

        elastic_gap = jnp.where(in_contact, rel_tangent - new_plastic, 0.0)
        interface_energy = jnp.sum(
            interface_weights
            * (
                0.5 * penalty_n * penetration**2
                + 0.5 * penalty_t * elastic_gap**2
            )
        )
        diagnostics = {
            "avg_tau": jnp.sum(interface_weights * tau) / total_interface_length,
            "avg_sigma_n": jnp.sum(interface_weights * normal_traction) / total_interface_length,
            "max_penetration": jnp.max(penetration),
            "max_slip": jnp.max(new_cum),
            "mu_eff_mean": jnp.sum(interface_weights * mu_eff) / total_interface_length,
            "interface_energy": interface_energy,
            "friction_strength": friction_strength,
            "friction_coefficient": mu_eff,
            "friction_velocity": friction_velocity,
            "rsf_state": new_rsf_state,
            "slip_rate": new_slip_rate,
        }
        return forces, new_plastic, new_cum, diagnostics

    def acceleration(
        u_flat: jax.Array,
        v_flat: jax.Array,
        plastic_slip: jax.Array,
        cum_slip: jax.Array,
        rsf_state: jax.Array,
        normal_scale: jax.Array,
        scheduled_shear_traction: jax.Array,
        actuator_displacement: jax.Array,
        tangential_friction_active: bool,
        zero_dofs: jax.Array,
        prescribed_dofs: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        elastic_energy, elastic_force = elastic_energy_and_force(u_flat)
        contact_force, plastic_new, cum_new, contact_diag = contact_response(
            u_flat,
            v_flat,
            plastic_slip,
            cum_slip,
            rsf_state,
            tangential_friction_active,
        )
        loading_face_displacement = jnp.mean(u_flat[moving_shear_loading_dofs])
        shear_traction = jnp.where(
            use_shear_spring,
            shear_loading_stiffness
            * (actuator_displacement - loading_face_displacement),
            scheduled_shear_traction,
        )
        force_ext = normal_scale * force_normal + shear_traction * force_shear_unit
        accel = zero_constrained_dofs(
            (force_ext - elastic_force - contact_force) / mass_flat,
            zero_dofs,
            prescribed_dofs,
        )
        diag = {
            "elastic_energy": elastic_energy,
            "kinetic_energy": jnp.array(0.0, dtype=dtype),
            **contact_diag,
            "applied_shear": shear_traction,
            "loading_face_displacement": loading_face_displacement,
        }
        diag["plastic_slip"] = plastic_new
        diag["cum_slip"] = cum_new
        return accel, diag

    u0 = jnp.zeros(total_dofs, dtype=dtype)
    v0 = jnp.zeros(total_dofs, dtype=dtype)
    plastic0 = jnp.zeros(master_nodes.shape[0], dtype=dtype)
    cum0 = jnp.zeros(master_nodes.shape[0], dtype=dtype)
    rsf_state0 = jnp.broadcast_to(
        rsf_parameters["initial_state"], master_nodes.shape
    ).astype(dtype)
    a0, diag0 = acceleration(
        u0,
        v0,
        plastic0,
        cum0,
        rsf_state0,
        model["normal_schedule_pressure"][0],
        model["pressure_schedule"][0],
        model["shear_displacement_pressure"][0],
        not relax_tangential_contact_during_normal,
        normal_phase_zero_dofs,
        prescribed_dofs,
    )
    initial_prescribed_velocity = jnp.concatenate(
        [
            jnp.full(
                prescribed_normal_dofs.shape,
                model["normal_velocity_pressure"][0],
                dtype=dtype,
            ),
            jnp.full(
                prescribed_shear_dofs.shape,
                model["shear_velocity_pressure"][0],
                dtype=dtype,
            ),
        ]
    )
    v_half0 = apply_constraints(
        0.5 * dt * a0,
        normal_phase_zero_dofs,
        prescribed_dofs,
        initial_prescribed_velocity,
    )

    def step(
        carry: tuple[jax.Array, ...],
        loading: jax.Array,
        zero_dofs: jax.Array,
        prescribed_dofs: jax.Array,
        allow_loading_stop: bool,
        apply_normal_relaxation: bool,
    ) -> tuple[tuple[jax.Array, ...], jax.Array]:
        (
            u_flat,
            v_half,
            plastic_slip,
            cum_slip,
            rsf_state,
            slip_rate,
            time_now,
            loading_stopped,
            previous_shear_displacement,
        ) = carry
        normal_scale = loading[0]
        shear_traction = loading[1]
        loading_stop_reached = jnp.any(
            _shear_loading_stop_candidates(
                cum_slip,
                slip_rate,
                shear_loading_stop_mask,
                critical_slip_profile,
                shear_loading_stop_slip,
                shear_loading_stop_uses_critical_profile,
                shear_loading_stop_velocity,
            )
        )
        stop_now = (
            allow_loading_stop and stop_shear_loading and loading_stop_reached
        )
        loading_stopped_new = loading_stopped | stop_now
        applied_shear_displacement = jnp.where(
            loading_stopped_new,
            previous_shear_displacement,
            loading[4],
        )
        applied_shear_velocity = jnp.where(
            loading_stopped_new,
            jnp.asarray(0.0, dtype=dtype),
            loading[5],
        )
        prescribed_values = jnp.concatenate(
            [
                jnp.full(
                    prescribed_normal_dofs.shape,
                    loading[2],
                    dtype=dtype,
                ),
                jnp.full(
                    prescribed_shear_dofs.shape,
                    applied_shear_displacement,
                    dtype=dtype,
                ),
            ]
        )
        prescribed_velocities = jnp.concatenate(
            [
                jnp.full(
                    prescribed_normal_dofs.shape,
                    loading[3],
                    dtype=dtype,
                ),
                jnp.full(
                    prescribed_shear_dofs.shape,
                    applied_shear_velocity,
                    dtype=dtype,
                ),
            ]
        )
        u_new = apply_constraints(
            u_flat + dt * v_half,
            zero_dofs,
            prescribed_dofs,
            prescribed_values,
        )
        accel, diag = acceleration(
            u_new,
            v_half,
            plastic_slip,
            cum_slip,
            rsf_state,
            normal_scale,
            shear_traction,
            applied_shear_displacement,
            allow_loading_stop or not relax_tangential_contact_during_normal,
            zero_dofs,
            prescribed_dofs,
        )
        velocity_trial = v_half + dt * accel
        if apply_normal_relaxation:
            velocity_trial = normal_relaxation_factor * velocity_trial
        v_half_new = apply_constraints(
            velocity_trial,
            zero_dofs,
            prescribed_dofs,
            prescribed_velocities,
        )
        kinetic = 0.5 * jnp.sum(mass_flat * v_half_new**2)
        output = jnp.array(
            [
                time_now + dt,
                diag["applied_shear"],
                diag["avg_tau"],
                diag["avg_sigma_n"],
                diag["max_penetration"],
                diag["max_slip"],
                diag["mu_eff_mean"],
                diag["elastic_energy"],
                diag["interface_energy"],
                kinetic,
                applied_shear_displacement,
                loading_stopped_new.astype(dtype),
                diag["loading_face_displacement"],
            ],
            dtype=dtype,
        )
        return (
            u_new,
            v_half_new,
            diag["plastic_slip"],
            diag["cum_slip"],
            diag["rsf_state"],
            diag["slip_rate"],
            time_now + dt,
            loading_stopped_new,
            applied_shear_displacement,
        ), output

    def run_phase(
        carry: tuple[jax.Array, ...],
        schedule: jax.Array,
        zero_dofs: jax.Array,
        prescribed_dofs: jax.Array,
        allow_loading_stop: bool,
        apply_normal_relaxation: bool,
    ) -> tuple[tuple[jax.Array, ...], jax.Array]:
        return jax.jit(
            lambda c, s: jax.lax.scan(
                lambda carry_state, loading: step(
                    carry_state,
                    loading,
                    zero_dofs,
                    prescribed_dofs,
                    allow_loading_stop,
                    apply_normal_relaxation,
                ),
                c,
                s,
            )
        )(carry, schedule)

    carry0 = (
        u0,
        v_half0,
        diag0["plastic_slip"],
        diag0["cum_slip"],
        diag0["rsf_state"],
        diag0["slip_rate"],
        jnp.array(0.0, dtype=dtype),
        jnp.array(False),
        jnp.asarray(model["shear_displacement_pressure"][0], dtype=dtype),
    )
    carry1, hist_pressure = run_phase(
        carry0,
        jnp.stack(
            [
                model["normal_schedule_pressure"],
                model["pressure_schedule"],
                model["normal_displacement_pressure"],
                model["normal_velocity_pressure"],
                model["shear_displacement_pressure"],
                model["shear_velocity_pressure"],
            ],
            axis=1,
        ),
        normal_phase_zero_dofs,
        prescribed_dofs,
        False,
        normal_relaxation,
    )
    carry2, hist_shear = run_phase(
        carry1,
        jnp.stack(
            [
                model["normal_schedule_shear"],
                model["shear_schedule"],
                model["normal_displacement_shear"],
                model["normal_velocity_shear"],
                model["shear_displacement_shear"],
                model["shear_velocity_shear"],
            ],
            axis=1,
        ),
        shear_phase_zero_dofs,
        prescribed_dofs,
        True,
        False,
    )
    jax.block_until_ready(hist_shear)

    history = np.vstack([np.asarray(hist_pressure), np.asarray(hist_shear)])
    # Do not accumulate the clock in float32: multi-million-step runs otherwise
    # drift by a visible fraction of a millisecond in the saved plots.
    history[:, 0] = dt * np.arange(1, history.shape[0] + 1, dtype=np.float64)
    column_names = [
        "time",
        "applied_shear",
        "avg_tau",
        "avg_sigma_n",
        "max_penetration",
        "max_slip",
        "mu_eff_mean",
        "elastic_energy",
        "interface_energy",
        "kinetic_energy",
        "applied_shear_displacement",
        "shear_loading_stopped",
        "loading_face_displacement",
    ]
    (
        final_u,
        final_v,
        final_plastic,
        final_cum,
        final_rsf_state,
        final_slip_rate,
        _final_time,
        _loading_stopped,
        _applied_shear_displacement,
    ) = carry2
    final_time = dt * (model["pressure_steps"] + model["shear_steps"])
    stop_indices = np.flatnonzero(history[:, 11] > 0.5)
    loading_stop_time = (
        float(history[stop_indices[0], 0]) if stop_indices.size else None
    )
    loading_stop_time_in_shear = (
        loading_stop_time - model["pressure_time"]
        if loading_stop_time is not None
        else None
    )
    loading_stop_displacement = (
        float(history[stop_indices[0], 10]) if stop_indices.size else None
    )

    summary = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "dtype": str(np.asarray(history).dtype),
        "mesh_size": config.mesh_size,
        "cfl": model["cfl"],
        "contact_safety_factor": model["contact_safety_factor"],
        "time_step_override": model["time_step_override"],
        "dt": dt,
        "dt_stable_limit": model["dt_stable_limit"],
        "dt_bulk": model["dt_bulk"],
        "dt_contact": model["dt_contact"],
        "dt_stability_limiter": model["dt_stability_limiter"],
        "dt_limiter": model["dt_limiter"],
        "pressure_time": model["pressure_time"],
        "shear_time": model["shear_time"],
        "normal_ramp_time": model["normal_ramp_time"],
        "normal_ramp_steps": model["normal_ramp_steps"],
        "shear_ramp_time": model["shear_ramp_time"],
        "shear_ramp_shape": model["shear_ramp_shape"],
        "stop_shear_loading_on_rupture": model[
            "stop_shear_loading_on_rupture"
        ],
        "shear_loading_stop_slip": model["shear_loading_stop_slip"],
        "shear_loading_stop_velocity": model["shear_loading_stop_velocity"],
        "shear_loading_stop_uses_critical_profile": model[
            "shear_loading_stop_uses_critical_profile"
        ],
        "shear_loading_stop_min_y": model["shear_loading_stop_min_y"],
        "shear_loading_stop_max_y": model["shear_loading_stop_max_y"],
        "shear_loading_stop_time": loading_stop_time,
        "shear_loading_stop_time_in_shear": loading_stop_time_in_shear,
        "shear_loading_stop_displacement": loading_stop_displacement,
        "relax_tangential_contact_during_normal": model[
            "relax_tangential_contact_during_normal"
        ],
        "pressure_steps": model["pressure_steps"],
        "shear_steps": model["shear_steps"],
        "tau_k": model["tau_k"],
        "tau_s": model["tau_s"],
        "shear_scale": model["shear_scale"],
        "normal_stress": model["normal_stress"],
        "normal_loading_mode": normal_loading_mode,
        "shear_loading_mode": shear_loading_mode,
        "shear_loading_stiffness": model["shear_loading_stiffness"],
        "shear_force_boundary": model["shear_force_boundary"],
        "shear_displacement_boundary": model["shear_displacement_boundary"],
        "normal_displacement": model["normal_displacement"],
        "normal_displacement_estimate": model["normal_displacement_estimate"],
        "shear_displacement_k": model["shear_displacement_k"],
        "shear_displacement_s": model["shear_displacement_s"],
        "critical_slip": model["critical_slip"],
        "mu_k": model["mu_k"],
        "loading_edge_nucleation_length": model[
            "loading_edge_nucleation_length"
        ],
        "loading_edge_critical_slip": model["loading_edge_critical_slip"],
        "mu_s_start_fraction": model["mu_s_start_fraction"],
        "mu_s_end_fraction": model["mu_s_end_fraction"],
        "pw_length": model["pw_length"],
        "pw_mu_s_ratio": model["pw_mu_s_ratio"],
        "pw_transition_length": model["pw_transition_length"],
        "leading_edge_guard_length": model["leading_edge_guard_length"],
        "leading_edge_guard_mu_s_ratio": model[
            "leading_edge_guard_mu_s_ratio"
        ],
        "leading_edge_guard_transition_length": model[
            "leading_edge_guard_transition_length"
        ],
        "leading_edge_tangential_taper_length": model[
            "leading_edge_tangential_taper_length"
        ],
        "leading_edge_tangential_plateau_length": model[
            "leading_edge_tangential_plateau_length"
        ],
        "leading_edge_tangential_taper_ratio": model[
            "leading_edge_tangential_taper_ratio"
        ],
        "friction_law": model["friction_law"],
        "rsf_reference_friction": model["rsf_reference_friction"],
        "rsf_direct_effect": model["rsf_direct_effect"],
        "rsf_state_effect": model["rsf_state_effect"],
        "rsf_reference_velocity": model["rsf_reference_velocity"],
        "rsf_reference_state": model["rsf_reference_state"],
        "rsf_characteristic_slip": model["rsf_characteristic_slip"],
        "rsf_initial_state": model["rsf_initial_state"],
        "leading_edge_creep_length": model["leading_edge_creep_length"],
        "leading_edge_creep_transition_length": model[
            "leading_edge_creep_transition_length"
        ],
        "leading_edge_creep_mu": model["leading_edge_creep_mu"],
        "leading_edge_creep_mu_k": model["leading_edge_creep_mu_k"],
        "leading_edge_creep_relaxation_time": model[
            "leading_edge_creep_relaxation_time"
        ],
        "normal_penalty": float(penalty_n),
        "tangential_penalty": model["tangential_penalty"],
        "lock_shear_edge_during_normal": bool(config.lock_shear_edge_during_normal),
        "moving_nodes": n_moving,
        "stationary_nodes": n_stationary,
        "moving_elements": int(moving.mesh.elements.shape[0]),
        "stationary_elements": int(stationary.mesh.elements.shape[0]),
        "final_time": float(final_time),
        "final_max_slip": float(jnp.max(final_cum)),
        "final_max_penetration": float(history[-1, 4]),
        "final_avg_tau": float(history[-1, 2]),
        "final_avg_sigma_n": float(history[-1, 3]),
        "legacy": {
            "materials": {name: asdict(mat) for name, mat in case.materials.items()},
            "friction": asdict(case.friction),
            "effective_friction": asdict(model["friction"]),
            "simulation": asdict(case.simulation),
            "geometry": {
                "moving": asdict(case.moving),
                "stationary": asdict(case.stationary),
            },
        },
    }

    if config.output_prefix:
        prefix = Path(config.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            prefix.with_suffix(".npz"),
            history=history,
            columns=np.asarray(column_names, dtype=object),
            final_u=np.asarray(final_u),
            final_v_half=np.asarray(final_v),
            final_plastic_slip=np.asarray(final_plastic),
            final_cum_slip=np.asarray(final_cum),
            final_rsf_state=np.asarray(final_rsf_state),
            final_slip_rate=np.asarray(final_slip_rate),
        )
        prefix.with_suffix(".json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    return {"summary": summary, "history": history, "columns": column_names}


def run_simulation_dumped(
    case: LegacyCase,
    config: RunConfig,
    data_path: Path,
    *,
    frames_per_phase: int = 2400,
    shear_frames_per_phase: int | None = None,
    interface_frames_per_phase: int | None = None,
    shear_interface_frames_per_phase: int | None = None,
    compression: str = "lzf",
    include_initial_frame: bool = True,
    checkpoint_path: Path | None = None,
    checkpoint_interval_seconds: float | None = None,
    checkpoint_deadline_monotonic: float | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    import h5py

    checkpoint_path = (
        None if checkpoint_path is None else checkpoint_path.expanduser().resolve()
    )
    if resume and checkpoint_path is None:
        raise ValueError("resume=True requires checkpoint_path.")
    if checkpoint_deadline_monotonic is not None and checkpoint_path is None:
        raise ValueError("checkpoint_deadline_monotonic requires checkpoint_path.")
    if checkpoint_interval_seconds is not None and checkpoint_interval_seconds <= 0.0:
        raise ValueError("checkpoint_interval_seconds must be positive.")
    checkpoint_requested = {"value": False}
    previous_signal_handlers: dict[int, Any] = {}
    if checkpoint_path is not None:
        def request_checkpoint(_signum: int, _frame: Any) -> None:
            checkpoint_requested["value"] = True

        for signum in (signal.SIGUSR1, signal.SIGTERM):
            previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_checkpoint)

    def restore_signal_handlers() -> None:
        for signum, handler in previous_signal_handlers.items():
            signal.signal(signum, handler)

    model = build_case_model(case, config)

    dtype = model["dtype"]
    moving = model["moving"]
    stationary = model["stationary"]
    moving_material = model["moving_material"]
    stationary_material = model["stationary_material"]
    base_fixed_dofs = model["fixed_dofs"]
    moving_normal_edge_dofs = model["moving_normal_edge_dofs"]
    moving_shear_edge_dofs = model["moving_shear_edge_dofs"]
    moving_shear_loading_dofs = model["moving_shear_loading_dofs"]
    force_normal = model["force_normal"]
    force_shear_unit = model["force_shear_unit"]
    mass_flat = model["mass_flat"]
    master_nodes = model["master_nodes"]
    slave_nodes = model["slave_nodes"]
    interface_weights = model["interface_weights"]
    mu_s_profile = model["mu_s_profile"]
    mu_k_profile = model["mu_k_profile"]
    critical_slip_profile = model["critical_slip_profile"]
    creep_weight_profile = model["creep_weight_profile"]
    creep_relaxation_time = jnp.asarray(
        model["leading_edge_creep_relaxation_time"], dtype=dtype
    )
    use_rate_state = model["friction_law"] in {
        "rate-state-vws",
        "rate-state-regularized",
    }
    use_regularized_rate_state = (
        model["friction_law"] == "rate-state-regularized"
    )
    rsf_parameters = {
        name: jnp.broadcast_to(jnp.asarray(value, dtype=dtype), master_nodes.shape)
        for name, value in model["rsf_parameters"].items()
    }
    penalty_n = model["penalty_n"]
    penalty_t = model["penalty_t"]
    dt = model["dt"]
    moving_offset = model["moving_offset"]
    total_dofs = model["total_dofs"]
    dimension = int(model["dimension"])
    interface_plot_master_nodes = model["interface_plot_master_nodes"]
    normal_loading_mode = str(model["normal_loading_mode"])
    use_normal_displacement = normal_loading_mode == "displacement"
    shear_loading_mode = str(model["shear_loading_mode"])
    use_shear_displacement = shear_loading_mode == "displacement"
    use_shear_spring = shear_loading_mode == "spring-displacement"
    shear_loading_stiffness = jnp.asarray(
        0.0
        if model["shear_loading_stiffness"] is None
        else model["shear_loading_stiffness"],
        dtype=dtype,
    )
    stop_shear_loading = bool(model["stop_shear_loading_on_rupture"])
    shear_loading_stop_slip = jnp.asarray(
        model["shear_loading_stop_slip"], dtype=dtype
    )
    shear_loading_stop_velocity = (
        None
        if model["shear_loading_stop_velocity"] is None
        else jnp.asarray(model["shear_loading_stop_velocity"], dtype=dtype)
    )
    shear_loading_stop_uses_critical_profile = bool(
        model["shear_loading_stop_uses_critical_profile"]
    )
    shear_loading_stop_mask = model["shear_loading_stop_mask"]
    relax_tangential_contact_during_normal = bool(
        model["relax_tangential_contact_during_normal"]
    )
    quasistatic_preloading = model["quasistatic_shear_fraction"] > 0.0
    normal_relaxation = model["normal_relaxation_time"] is not None
    normal_relaxation_factor = jnp.asarray(
        (
            1.0
            if not normal_relaxation
            else math.exp(-dt / model["normal_relaxation_time"])
        ),
        dtype=dtype,
    )

    n_moving = moving.n_nodes
    n_stationary = stationary.n_nodes
    friction = model["friction"]

    if config.lock_shear_edge_during_normal:
        normal_phase_zero_dofs = jnp.unique(
            jnp.concatenate([base_fixed_dofs, moving_shear_edge_dofs])
        )
    else:
        normal_phase_zero_dofs = base_fixed_dofs
    shear_phase_zero_dofs = base_fixed_dofs
    prescribed_normal_dofs = (
        moving_normal_edge_dofs
        if use_normal_displacement
        else jnp.zeros((0,), dtype=jnp.int32)
    )
    prescribed_shear_dofs = (
        moving_shear_loading_dofs
        if use_shear_displacement
        else jnp.zeros((0,), dtype=jnp.int32)
    )
    prescribed_dofs = jnp.concatenate([prescribed_normal_dofs, prescribed_shear_dofs])

    def apply_constraints(
        vec: jax.Array,
        zero_dofs: jax.Array,
        prescribed_dofs: jax.Array,
        prescribed_values: jax.Array,
    ) -> jax.Array:
        constrained = vec.at[zero_dofs].set(0.0)
        return constrained.at[prescribed_dofs].set(prescribed_values)

    def zero_constrained_dofs(
        vec: jax.Array,
        zero_dofs: jax.Array,
        prescribed_dofs: jax.Array,
    ) -> jax.Array:
        constrained = vec.at[zero_dofs].set(0.0)
        return constrained.at[prescribed_dofs].set(0.0)

    def split_flat(flat: jax.Array, n_nodes: int) -> jax.Array:
        return flat.reshape(n_nodes, dimension)

    def split_u(u_flat: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (
            split_flat(u_flat[: dimension * n_moving], n_moving),
            split_flat(u_flat[dimension * n_moving :], n_stationary),
        )

    def compute_strain(grad_u: jax.Array) -> jax.Array:
        return 0.5 * (grad_u + jnp.swapaxes(grad_u, -1, -2))

    def compute_stress(eps: jax.Array, mat: LegacyMaterial) -> jax.Array:
        return 2.0 * mat.mu * eps + mat.lmbda * jnp.trace(
            eps, axis1=-2, axis2=-1
        )[..., None, None] * jnp.eye(dimension, dtype=eps.dtype)

    moving_integration_weights = moving.operator.get_integration_weights()
    stationary_integration_weights = stationary.operator.get_integration_weights()

    def elastic_energy_total(u_flat: jax.Array) -> jax.Array:
        u_moving, u_stationary = split_u(u_flat)
        eps_moving = compute_strain(moving.operator.grad(u_moving))
        eps_stationary = compute_strain(stationary.operator.grad(u_stationary))
        return moving.operator.integrate(
            moving_material.mu * jnp.einsum("...ij,...ij->...", eps_moving, eps_moving)
            + 0.5
            * moving_material.lmbda
            * jnp.trace(eps_moving, axis1=-2, axis2=-1) ** 2
        ) + stationary.operator.integrate(
            stationary_material.mu
            * jnp.einsum("...ij,...ij->...", eps_stationary, eps_stationary)
            + 0.5
            * stationary_material.lmbda
            * jnp.trace(eps_stationary, axis1=-2, axis2=-1) ** 2
        )

    elastic_energy_and_force = jax.jit(jax.value_and_grad(elastic_energy_total))

    total_interface_length = jnp.sum(interface_weights)
    moving_iface_x = dimension * master_nodes
    moving_iface_y = dimension * master_nodes + 1
    stationary_iface_x = moving_offset + dimension * slave_nodes
    stationary_iface_y = moving_offset + dimension * slave_nodes + 1
    inverse_mass_moving_y = 1.0 / mass_flat[moving_iface_y]
    inverse_mass_stationary_y = 1.0 / mass_flat[stationary_iface_y]
    inverse_mass_sum_y = inverse_mass_moving_y + inverse_mass_stationary_y
    moving_velocity_share = inverse_mass_moving_y / inverse_mass_sum_y
    stationary_velocity_share = inverse_mass_stationary_y / inverse_mass_sum_y
    relative_impulse_factor = dt * interface_weights * inverse_mass_sum_y

    def apply_regularized_friction(
        u_flat: jax.Array,
        v_flat: jax.Array,
        rsf_state: jax.Array,
        tangential_friction_active: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        u_moving, u_stationary = split_u(u_flat)
        rel_normal = u_moving[master_nodes, 0] - u_stationary[slave_nodes, 0]
        penetration = jnp.maximum(rel_normal, 0.0)
        in_contact = penetration > 0.0
        normal_traction = penalty_n * penetration
        free_relative_velocity = (
            v_flat[moving_iface_y] - v_flat[stationary_iface_y]
        )
        active = in_contact & tangential_friction_active
        corrected_velocity, strength = project_regularized_rate_state_velocity(
            free_relative_velocity,
            normal_traction,
            rsf_state,
            jnp.where(active, relative_impulse_factor, 0.0),
            reference_friction=rsf_parameters["reference_friction"],
            direct_effect=rsf_parameters["direct_effect"],
            state_effect=rsf_parameters["state_effect"],
            reference_velocity=rsf_parameters["reference_velocity"],
            characteristic_slip=rsf_parameters["characteristic_slip"],
        )
        corrected_velocity = jnp.where(
            active, corrected_velocity, free_relative_velocity
        )
        strength = jnp.where(active, strength, 0.0)
        relative_correction = free_relative_velocity - corrected_velocity
        projected = v_flat.at[moving_iface_y].add(
            -moving_velocity_share * relative_correction
        )
        projected = projected.at[stationary_iface_y].add(
            stationary_velocity_share * relative_correction
        )
        speed = jnp.where(active, jnp.abs(corrected_velocity), 0.0)
        coefficient = jnp.where(
            normal_traction > 0.0,
            strength / jnp.maximum(normal_traction, jnp.finfo(dtype).tiny),
            0.0,
        )
        signed_strength = jnp.sign(corrected_velocity) * strength
        return projected, corrected_velocity, speed, coefficient, signed_strength

    def contact_response(
        u_flat: jax.Array,
        v_flat: jax.Array,
        plastic_slip: jax.Array,
        cum_slip: jax.Array,
        rsf_state: jax.Array,
        tangential_friction_active: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        u_moving, u_stationary = split_u(u_flat)
        v_moving, v_stationary = split_u(v_flat)
        rel_normal = u_moving[master_nodes, 0] - u_stationary[slave_nodes, 0]
        penetration = jnp.maximum(rel_normal, 0.0)
        in_contact = penetration > 0.0

        rel_tangent = u_moving[master_nodes, 1] - u_stationary[slave_nodes, 1]
        trial_tau = penalty_t * (rel_tangent - plastic_slip)
        normal_traction = penalty_n * penetration
        slip_weakening_mu = jnp.maximum(
            mu_k_profile,
            mu_s_profile
            - (mu_s_profile - mu_k_profile)
            * jnp.minimum(cum_slip / critical_slip_profile, 1.0),
        )
        friction_velocity = jnp.abs(
            v_moving[master_nodes, 1] - v_stationary[slave_nodes, 1]
        )
        if use_regularized_rate_state:
            new_rsf_state = rsf_state
            friction_strength = regularized_rate_state_strength(
                friction_velocity,
                normal_traction,
                new_rsf_state,
                reference_friction=rsf_parameters["reference_friction"],
                direct_effect=rsf_parameters["direct_effect"],
                state_effect=rsf_parameters["state_effect"],
                reference_velocity=rsf_parameters["reference_velocity"],
                characteristic_slip=rsf_parameters["characteristic_slip"],
            )
            mu_eff = jnp.where(
                normal_traction > 0.0,
                friction_strength
                / jnp.maximum(normal_traction, jnp.finfo(dtype).tiny),
                0.0,
            )
        elif use_rate_state:
            evolved_rsf_state = update_ageing_state(
                rsf_state,
                friction_velocity,
                dt,
                rsf_parameters["characteristic_slip"],
            )
            new_rsf_state = jnp.where(
                in_contact, evolved_rsf_state, rsf_state
            )
            mu_eff = velocity_weakening_strengthening_coefficient(
                friction_velocity,
                new_rsf_state,
                reference_friction=rsf_parameters["reference_friction"],
                direct_effect=rsf_parameters["direct_effect"],
                state_effect=rsf_parameters["state_effect"],
                reference_velocity=rsf_parameters["reference_velocity"],
                reference_state=rsf_parameters["reference_state"],
            )
        else:
            new_rsf_state = rsf_state
            mu_eff = slip_weakening_mu
        if not use_regularized_rate_state:
            friction_strength = mu_eff * normal_traction
        friction_strength = jnp.where(in_contact, friction_strength, 0.0)
        if use_regularized_rate_state:
            new_plastic = plastic_slip
            new_cum = cum_slip
            new_slip_rate = friction_velocity
            tau = jnp.zeros_like(trial_tau)
        else:
            yield_tau = friction_strength
            sliding = in_contact & (jnp.abs(trial_tau) > yield_tau)
            plastic_correction = (
                jnp.sign(trial_tau)
                * jnp.maximum(jnp.abs(trial_tau) - yield_tau, 0.0)
                / penalty_t
            )
            viscous_fraction = dt / (dt + creep_relaxation_time)
            correction_fraction = 1.0 - creep_weight_profile * (
                1.0 - viscous_fraction
            )
            plastic_increment = jnp.where(
                sliding, correction_fraction * plastic_correction, 0.0
            )
            new_plastic = plastic_slip + plastic_increment
            tau = jnp.where(
                in_contact,
                trial_tau - penalty_t * plastic_increment,
                0.0,
            )
            new_cum = cum_slip + jnp.abs(plastic_increment)
            new_slip_rate = jnp.abs(plastic_increment) / dt
            tau = jnp.where(tangential_friction_active, tau, 0.0)
            new_plastic = jnp.where(
                tangential_friction_active, new_plastic, rel_tangent
            )
            new_cum = jnp.where(tangential_friction_active, new_cum, cum_slip)
            new_slip_rate = jnp.where(
                tangential_friction_active, new_slip_rate, 0.0
            )

        forces = jnp.zeros(total_dofs, dtype=dtype)
        forces = forces.at[moving_iface_x].add(interface_weights * normal_traction)
        forces = forces.at[stationary_iface_x].add(-interface_weights * normal_traction)
        forces = forces.at[moving_iface_y].add(interface_weights * tau)
        forces = forces.at[stationary_iface_y].add(-interface_weights * tau)

        elastic_gap = jnp.where(in_contact, rel_tangent - new_plastic, 0.0)
        tangential_energy = jnp.where(
            use_regularized_rate_state,
            0.0,
            0.5 * penalty_t * elastic_gap**2,
        )
        interface_energy = jnp.sum(
            interface_weights
            * (
                0.5 * penalty_n * penetration**2
                + tangential_energy
            )
        )
        diagnostics = {
            "avg_tau": jnp.sum(interface_weights * tau) / total_interface_length,
            "avg_sigma_n": jnp.sum(interface_weights * normal_traction)
            / total_interface_length,
            "max_penetration": jnp.max(penetration),
            "max_slip": jnp.max(new_cum),
            "mu_eff_mean": jnp.sum(interface_weights * mu_eff) / total_interface_length,
            "interface_energy": interface_energy,
            "friction_strength": friction_strength,
            "friction_coefficient": mu_eff,
            "friction_velocity": friction_velocity,
            "rsf_state": new_rsf_state,
            "slip_rate": new_slip_rate,
        }
        return forces, new_plastic, new_cum, diagnostics

    def acceleration(
        u_flat: jax.Array,
        v_flat: jax.Array,
        plastic_slip: jax.Array,
        cum_slip: jax.Array,
        rsf_state: jax.Array,
        normal_scale: jax.Array,
        scheduled_shear_traction: jax.Array,
        actuator_displacement: jax.Array,
        tangential_friction_active: bool,
        zero_dofs: jax.Array,
        prescribed_dofs: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        elastic_energy, elastic_force = elastic_energy_and_force(u_flat)
        contact_force, plastic_new, cum_new, contact_diag = contact_response(
            u_flat,
            v_flat,
            plastic_slip,
            cum_slip,
            rsf_state,
            tangential_friction_active,
        )
        loading_face_displacement = jnp.mean(u_flat[moving_shear_loading_dofs])
        shear_traction = jnp.where(
            use_shear_spring,
            shear_loading_stiffness
            * (actuator_displacement - loading_face_displacement),
            scheduled_shear_traction,
        )
        force_ext = normal_scale * force_normal + shear_traction * force_shear_unit
        accel = zero_constrained_dofs(
            (force_ext - elastic_force - contact_force) / mass_flat,
            zero_dofs,
            prescribed_dofs,
        )
        diag = {
            "elastic_energy": elastic_energy,
            "kinetic_energy": jnp.array(0.0, dtype=dtype),
            **contact_diag,
            "applied_shear": shear_traction,
            "loading_face_displacement": loading_face_displacement,
        }
        diag["plastic_slip"] = plastic_new
        diag["cum_slip"] = cum_new
        return accel, diag

    def make_row(
        time_now: jax.Array,
        shear_traction: jax.Array,
        diag: dict[str, jax.Array],
        kinetic: jax.Array,
        applied_shear_displacement: jax.Array,
        loading_stopped: jax.Array,
    ) -> jax.Array:
        return jnp.array(
            [
                time_now,
                shear_traction,
                diag["avg_tau"],
                diag["avg_sigma_n"],
                diag["max_penetration"],
                diag["max_slip"],
                diag["mu_eff_mean"],
                diag["elastic_energy"],
                diag["interface_energy"],
                kinetic,
                applied_shear_displacement,
                loading_stopped.astype(dtype),
                diag["loading_face_displacement"],
            ],
            dtype=dtype,
        )

    u0 = jnp.zeros(total_dofs, dtype=dtype)
    v0 = jnp.zeros(total_dofs, dtype=dtype)
    plastic0 = jnp.zeros(master_nodes.shape[0], dtype=dtype)
    cum0 = jnp.zeros(master_nodes.shape[0], dtype=dtype)
    rsf_state0 = jnp.broadcast_to(
        rsf_parameters["initial_state"], master_nodes.shape
    ).astype(dtype)
    a0, diag0 = acceleration(
        u0,
        v0,
        plastic0,
        cum0,
        rsf_state0,
        model["normal_schedule_pressure"][0],
        model["pressure_schedule"][0],
        model["shear_displacement_pressure"][0],
        not relax_tangential_contact_during_normal,
        normal_phase_zero_dofs,
        prescribed_dofs,
    )
    initial_prescribed_velocity = jnp.concatenate(
        [
            jnp.full(
                prescribed_normal_dofs.shape,
                model["normal_velocity_pressure"][0],
                dtype=dtype,
            ),
            jnp.full(
                prescribed_shear_dofs.shape,
                model["shear_velocity_pressure"][0],
                dtype=dtype,
            ),
        ]
    )
    v_half0 = apply_constraints(
        0.5 * dt * a0,
        normal_phase_zero_dofs,
        prescribed_dofs,
        initial_prescribed_velocity,
    )

    def step(
        carry: tuple[jax.Array, ...],
        loading: jax.Array,
        zero_dofs: jax.Array,
        prescribed_dofs: jax.Array,
        allow_loading_stop: bool,
        apply_normal_relaxation: bool,
    ) -> tuple[tuple[jax.Array, ...], jax.Array]:
        (
            u_flat,
            v_half,
            plastic_slip,
            cum_slip,
            rsf_state,
            slip_rate,
            friction_coefficient,
            friction_velocity,
            friction_strength,
            time_now,
            loading_stopped,
            previous_shear_displacement,
        ) = carry
        normal_scale = loading[0]
        shear_traction = loading[1]
        loading_stop_reached = jnp.any(
            _shear_loading_stop_candidates(
                cum_slip,
                slip_rate,
                shear_loading_stop_mask,
                critical_slip_profile,
                shear_loading_stop_slip,
                shear_loading_stop_uses_critical_profile,
                shear_loading_stop_velocity,
            )
        )
        stop_now = (
            allow_loading_stop and stop_shear_loading and loading_stop_reached
        )
        loading_stopped_new = loading_stopped | stop_now
        applied_shear_displacement = jnp.where(
            loading_stopped_new,
            previous_shear_displacement,
            loading[4],
        )
        applied_shear_velocity = jnp.where(
            loading_stopped_new,
            jnp.asarray(0.0, dtype=dtype),
            loading[5],
        )
        prescribed_values = jnp.concatenate(
            [
                jnp.full(
                    prescribed_normal_dofs.shape,
                    loading[2],
                    dtype=dtype,
                ),
                jnp.full(
                    prescribed_shear_dofs.shape,
                    applied_shear_displacement,
                    dtype=dtype,
                ),
            ]
        )
        prescribed_velocities = jnp.concatenate(
            [
                jnp.full(
                    prescribed_normal_dofs.shape,
                    loading[3],
                    dtype=dtype,
                ),
                jnp.full(
                    prescribed_shear_dofs.shape,
                    applied_shear_velocity,
                    dtype=dtype,
                ),
            ]
        )
        u_new = apply_constraints(
            u_flat + dt * v_half,
            zero_dofs,
            prescribed_dofs,
            prescribed_values,
        )
        tangential_friction_active = (
            allow_loading_stop or not relax_tangential_contact_during_normal
        )
        state_for_acceleration = rsf_state
        if use_regularized_rate_state:
            evolved_state = update_ageing_state(
                rsf_state,
                friction_velocity,
                dt,
                rsf_parameters["characteristic_slip"],
            )
            state_for_acceleration = jnp.where(
                tangential_friction_active, evolved_state, rsf_state
            )
        accel, diag = acceleration(
            u_new,
            v_half,
            plastic_slip,
            cum_slip,
            state_for_acceleration,
            normal_scale,
            shear_traction,
            applied_shear_displacement,
            tangential_friction_active,
            zero_dofs,
            prescribed_dofs,
        )
        velocity_trial = v_half + dt * accel
        if apply_normal_relaxation:
            velocity_trial = normal_relaxation_factor * velocity_trial
        v_half_new = apply_constraints(
            velocity_trial,
            zero_dofs,
            prescribed_dofs,
            prescribed_velocities,
        )
        if use_regularized_rate_state:
            (
                v_half_new,
                corrected_relative_velocity,
                corrected_speed,
                corrected_coefficient,
                signed_strength,
            ) = apply_regularized_friction(
                u_new,
                v_half_new,
                state_for_acceleration,
                tangential_friction_active,
            )
            v_half_new = apply_constraints(
                v_half_new,
                zero_dofs,
                prescribed_dofs,
                prescribed_velocities,
            )
            plastic_new = jnp.where(
                tangential_friction_active,
                plastic_slip + dt * corrected_relative_velocity,
                plastic_slip,
            )
            cumulative_new = cum_slip + dt * corrected_speed
            strength_magnitude = jnp.abs(signed_strength)
            diag["plastic_slip"] = plastic_new
            diag["cum_slip"] = cumulative_new
            diag["rsf_state"] = state_for_acceleration
            diag["slip_rate"] = corrected_speed
            diag["friction_velocity"] = corrected_speed
            diag["friction_coefficient"] = corrected_coefficient
            diag["friction_strength"] = strength_magnitude
            diag["max_slip"] = jnp.max(cumulative_new)
            diag["avg_tau"] = (
                jnp.sum(interface_weights * signed_strength)
                / total_interface_length
            )
            diag["mu_eff_mean"] = (
                jnp.sum(interface_weights * corrected_coefficient)
                / total_interface_length
            )
        kinetic = 0.5 * jnp.sum(mass_flat * v_half_new**2)
        output = make_row(
            time_now + dt,
            diag["applied_shear"],
            diag,
            kinetic,
            applied_shear_displacement,
            loading_stopped_new,
        )
        return (
            u_new,
            v_half_new,
            diag["plastic_slip"],
            diag["cum_slip"],
            diag["rsf_state"],
            diag["slip_rate"],
            diag["friction_coefficient"],
            diag["friction_velocity"],
            diag["friction_strength"],
            time_now + dt,
            loading_stopped_new,
            applied_shear_displacement,
        ), output

    @jax.jit
    def observe_fields(
        u_flat: jax.Array, v_flat: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        u_moving, u_stationary = split_u(u_flat)
        v_moving, v_stationary = split_u(v_flat)
        eps_moving = quadrature_weighted_element_average(
            compute_strain(moving.operator.grad(u_moving)),
            moving_integration_weights,
        )
        eps_stationary = quadrature_weighted_element_average(
            compute_strain(stationary.operator.grad(u_stationary)),
            stationary_integration_weights,
        )
        sigma_moving = compute_stress(eps_moving, moving_material)
        sigma_stationary = compute_stress(eps_stationary, stationary_material)
        return (
            u_moving,
            u_stationary,
            v_moving,
            v_stationary,
            eps_moving,
            eps_stationary,
            sigma_moving,
            sigma_stationary,
        )

    chunk_runners: dict[tuple[str, int], Any] = {}

    def advance_chunk(
        carry: tuple[jax.Array, ...],
        schedule_chunk: jax.Array,
        zero_dofs: jax.Array,
        prescribed_dofs: jax.Array,
        phase_label: str,
    ) -> tuple[tuple[jax.Array, ...], jax.Array]:
        length = int(schedule_chunk.shape[0])
        key = (phase_label, length)
        if key not in chunk_runners:
            chunk_runners[key] = jax.jit(
                lambda c, s: jax.lax.scan(
                    lambda carry_state, loading: step(
                        carry_state,
                        loading,
                        zero_dofs,
                        prescribed_dofs,
                        phase_label == "shear",
                        phase_label == "normal" and normal_relaxation,
                    ),
                    c,
                    s,
                )
            )
        return chunk_runners[key](carry, schedule_chunk)

    def sample_stop_indices(phase_steps: int, target_frames: int) -> np.ndarray:
        phase_steps = max(1, int(phase_steps))
        target_frames = max(1, int(target_frames))
        n_samples = min(phase_steps, target_frames)
        if n_samples == phase_steps:
            return np.arange(1, phase_steps + 1, dtype=np.int32)
        return np.floor(
            np.linspace(1, phase_steps, num=n_samples, endpoint=True, dtype=np.float64)
        ).astype(np.int32)

    shear_frames_per_phase = (
        frames_per_phase if shear_frames_per_phase is None else shear_frames_per_phase
    )
    pressure_sample_stops = sample_stop_indices(model["pressure_steps"], frames_per_phase)
    shear_sample_stops = sample_stop_indices(model["shear_steps"], shear_frames_per_phase)
    separate_interface_output = (
        interface_frames_per_phase is not None
        or shear_interface_frames_per_phase is not None
    )
    interface_frames_per_phase = (
        frames_per_phase
        if interface_frames_per_phase is None
        else interface_frames_per_phase
    )
    shear_interface_frames_per_phase = (
        shear_frames_per_phase
        if shear_interface_frames_per_phase is None
        else shear_interface_frames_per_phase
    )
    pressure_interface_stops = sample_stop_indices(
        model["pressure_steps"], interface_frames_per_phase
    )
    shear_interface_stops = sample_stop_indices(
        model["shear_steps"], shear_interface_frames_per_phase
    )
    pressure_chunk_sizes = np.diff(np.concatenate(([0], pressure_sample_stops)))
    shear_chunk_sizes = np.diff(np.concatenate(([0], shear_sample_stops)))
    save_every_pressure = int(np.median(pressure_chunk_sizes))
    save_every_shear = int(np.median(shear_chunk_sizes))
    n_press_frames = int(pressure_sample_stops.shape[0])
    n_shear_frames = int(shear_sample_stops.shape[0])
    total_frames = (1 if include_initial_frame else 0) + n_press_frames + n_shear_frames
    n_press_interface_frames = int(pressure_interface_stops.shape[0])
    n_shear_interface_frames = int(shear_interface_stops.shape[0])
    total_interface_frames = (
        (1 if include_initial_frame else 0)
        + n_press_interface_frames
        + n_shear_interface_frames
    )

    history_columns = [
        "time",
        "applied_shear",
        "avg_tau",
        "avg_sigma_n",
        "max_penetration",
        "max_slip",
        "mu_eff_mean",
        "elastic_energy",
        "interface_energy",
        "kinetic_energy",
        "applied_shear_displacement",
        "shear_loading_stopped",
        "loading_face_displacement",
    ]

    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    interface_frame_count = 0

    def _create_group_datasets(h5: h5py.File, block: BlockModel, name: str, n_frames: int, n_nodes: int, n_elem: int):
        grp = h5.create_group(name)
        grp.create_dataset("coords", data=np.asarray(block.mesh.coords))
        grp.create_dataset("elements", data=np.asarray(block.mesh.elements))
        grp.create_dataset("plot_elements", data=np.asarray(block.plot_elements))
        grp.create_dataset("plot_parent_elements", data=np.asarray(block.plot_parent_elements))
        kwargs = dict(compression=compression, chunks=(1, n_nodes, dimension))
        grp.create_dataset("displacement", shape=(n_frames, n_nodes, dimension), dtype="f4", **kwargs)
        grp.create_dataset("velocity", shape=(n_frames, n_nodes, dimension), dtype="f4", **kwargs)
        elem_kwargs = dict(compression=compression, chunks=(1, n_elem, dimension, dimension))
        grp.create_dataset("strain", shape=(n_frames, n_elem, dimension, dimension), dtype="f4", **elem_kwargs)
        grp.create_dataset("stress", shape=(n_frames, n_elem, dimension, dimension), dtype="f4", **elem_kwargs)
        return grp

    def save_frame(
        h5: h5py.File,
        frame_idx: int,
        carry: tuple[jax.Array, ...],
        history_row: np.ndarray,
        *,
        phase_id: int,
        step_id: int,
    ) -> None:
        (
            u_flat,
            v_half,
            plastic_slip,
            cum_slip,
            rsf_state,
            slip_rate,
            friction_coefficient,
            friction_velocity,
            friction_strength,
            _time_now,
            _loading_stopped,
            _applied_shear_displacement,
        ) = carry
        (
            u_moving,
            u_stationary,
            v_moving,
            v_stationary,
            eps_moving,
            eps_stationary,
            sigma_moving,
            sigma_stationary,
        ) = observe_fields(u_flat, v_half)
        h5["moving/displacement"][frame_idx] = np.asarray(u_moving, dtype=np.float32)
        h5["moving/velocity"][frame_idx] = np.asarray(v_moving, dtype=np.float32)
        h5["moving/strain"][frame_idx] = np.asarray(eps_moving, dtype=np.float32)
        h5["moving/stress"][frame_idx] = np.asarray(sigma_moving, dtype=np.float32)
        h5["stationary/displacement"][frame_idx] = np.asarray(u_stationary, dtype=np.float32)
        h5["stationary/velocity"][frame_idx] = np.asarray(v_stationary, dtype=np.float32)
        h5["stationary/strain"][frame_idx] = np.asarray(eps_stationary, dtype=np.float32)
        h5["stationary/stress"][frame_idx] = np.asarray(sigma_stationary, dtype=np.float32)
        # The simulation state uses float32, but the output clock must come from
        # its integer step count to avoid cumulative float32 timing drift.
        absolute_step = step_id if phase_id != 2 else model["pressure_steps"] + step_id
        history_row = np.asarray(history_row, dtype=np.float32).copy()
        history_row[0] = absolute_step * dt
        h5["history"][frame_idx] = history_row
        h5["phase_id"][frame_idx] = phase_id
        h5["step_id"][frame_idx] = step_id
        h5["interface/plastic_slip"][frame_idx] = np.asarray(
            plastic_slip[interface_plot_mask], dtype=np.float32
        )
        h5["interface/cumulative_slip"][frame_idx] = np.asarray(
            cum_slip[interface_plot_mask], dtype=np.float32
        )
        h5["interface/rsf_state"][frame_idx] = np.asarray(
            rsf_state[interface_plot_mask], dtype=np.float32
        )
        h5["interface/friction_coefficient"][frame_idx] = np.asarray(
            friction_coefficient[interface_plot_mask], dtype=np.float32
        )
        h5["interface/friction_velocity"][frame_idx] = np.asarray(
            friction_velocity[interface_plot_mask], dtype=np.float32
        )
        h5["interface/friction_strength"][frame_idx] = np.asarray(
            friction_strength[interface_plot_mask], dtype=np.float32
        )
        h5["interface/slip_rate"][frame_idx] = np.asarray(
            slip_rate[interface_plot_mask], dtype=np.float32
        )

    def save_high_rate_interface(
        h5: h5py.File,
        frame_idx: int,
        carry: tuple[jax.Array, ...],
        history_row: np.ndarray,
        *,
        phase_id: int,
        step_id: int,
    ) -> None:
        (
            _u_flat,
            _v_half,
            plastic_slip,
            cum_slip,
            rsf_state,
            slip_rate,
            friction_coefficient,
            friction_velocity,
            friction_strength,
            _time_now,
            _loading_stopped,
            _applied_shear_displacement,
        ) = carry
        absolute_step = step_id if phase_id != 2 else model["pressure_steps"] + step_id
        history_row = np.asarray(history_row, dtype=np.float32).copy()
        history_row[0] = absolute_step * dt
        group = h5["interface_high_rate"]
        group["history"][frame_idx] = history_row
        group["phase_id"][frame_idx] = phase_id
        group["step_id"][frame_idx] = step_id
        values = {
            "plastic_slip": plastic_slip,
            "cumulative_slip": cum_slip,
            "rsf_state": rsf_state,
            "friction_coefficient": friction_coefficient,
            "friction_velocity": friction_velocity,
            "friction_strength": friction_strength,
            "slip_rate": slip_rate,
        }
        for name, value in values.items():
            group[name][frame_idx] = np.asarray(
                value[interface_plot_mask], dtype=np.float32
            )

    kinetic0 = 0.5 * jnp.sum(mass_flat * v_half0**2)
    initial_row = np.asarray(
        make_row(
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
            diag0,
            kinetic0,
            jnp.asarray(model["shear_displacement_pressure"][0], dtype=dtype),
            jnp.array(False),
        )
    )
    carry = (
        u0,
        v_half0,
        diag0["plastic_slip"],
        diag0["cum_slip"],
        diag0["rsf_state"],
        diag0["slip_rate"],
        diag0["friction_coefficient"],
        diag0["friction_velocity"],
        diag0["friction_strength"],
        jnp.asarray(0.0, dtype=dtype),
        jnp.array(False),
        jnp.asarray(model["shear_displacement_pressure"][0], dtype=dtype),
    )
    resume_phase_id = 0
    resume_step_id = 0
    if resume:
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not data_path.exists():
            raise FileNotFoundError(f"Simulation dump not found: {data_path}")
        checkpoint_metadata, checkpoint_carry = _read_simulation_checkpoint(
            checkpoint_path
        )
        expected_metadata = {
            "version": 1,
            "carry_count": len(carry),
            "pressure_steps": int(model["pressure_steps"]),
            "shear_steps": int(model["shear_steps"]),
            "total_frames": total_frames,
            "total_interface_frames": total_interface_frames,
        }
        for name, expected in expected_metadata.items():
            if checkpoint_metadata.get(name) != expected:
                raise ValueError(
                    f"Checkpoint {name} mismatch: "
                    f"{checkpoint_metadata.get(name)!r} != {expected!r}."
                )
        if not math.isclose(
            float(checkpoint_metadata["dt"]), float(dt), rel_tol=1.0e-7
        ):
            raise ValueError("Checkpoint dt does not match the current case.")
        carry = tuple(jnp.asarray(value) for value in checkpoint_carry)
        frame_count = int(checkpoint_metadata["frame_count"])
        interface_frame_count = int(checkpoint_metadata["interface_frame_count"])
        resume_phase_id = int(checkpoint_metadata["phase_id"])
        resume_step_id = int(checkpoint_metadata["step_id"])

        # Temporarily move the existing datasets aside so the common metadata
        # initialization path can run without reallocating or copying old frames.
        with h5py.File(data_path, "r+") as existing_h5:
            if "_resume_stash" in existing_h5:
                for name in list(existing_h5.keys()):
                    if name != "_resume_stash":
                        del existing_h5[name]
                stash = existing_h5["_resume_stash"]
                for name in list(stash.keys()):
                    existing_h5.move(f"_resume_stash/{name}", name)
                del existing_h5["_resume_stash"]
            root_names = list(existing_h5.keys())
            existing_h5.create_group("_resume_stash")
            for name in root_names:
                existing_h5.move(name, f"_resume_stash/{name}")
            existing_h5.flush()

    last_checkpoint_time = time.monotonic()
    with h5py.File(data_path, "r+" if resume else "w") as h5:
        h5.attrs["backend"] = jax.default_backend()
        h5.attrs["cfl"] = model["cfl"]
        h5.attrs["contact_safety_factor"] = model["contact_safety_factor"]
        if model["time_step_override"] is not None:
            h5.attrs["time_step_override"] = model["time_step_override"]
        h5.attrs["dt"] = dt
        h5.attrs["dt_stable_limit"] = model["dt_stable_limit"]
        h5.attrs["dt_bulk"] = model["dt_bulk"]
        h5.attrs["dt_contact"] = model["dt_contact"]
        h5.attrs["dt_stability_limiter"] = model["dt_stability_limiter"]
        h5.attrs["dt_limiter"] = model["dt_limiter"]
        h5.attrs["save_every_pressure"] = save_every_pressure
        h5.attrs["save_every_shear"] = save_every_shear
        h5.attrs["pressure_steps"] = model["pressure_steps"]
        h5.attrs["shear_steps"] = model["shear_steps"]
        h5.attrs["pressure_frames_target"] = frames_per_phase
        h5.attrs["shear_frames_target"] = shear_frames_per_phase
        h5.attrs["interface_normal_frames_target"] = interface_frames_per_phase
        h5.attrs["interface_shear_frames_target"] = shear_interface_frames_per_phase
        h5.attrs["pressure_frames_actual"] = n_press_frames
        h5.attrs["shear_frames_actual"] = n_shear_frames
        h5.attrs["frame_sampling_mode"] = "linspace-stop-indices"
        h5.attrs["dimension"] = dimension
        h5.attrs["thickness"] = float(model["thickness"])
        h5.attrs["include_initial_frame"] = int(include_initial_frame)
        h5.attrs["normal_ramp_time"] = model["normal_ramp_time"]
        h5.attrs["normal_ramp_steps"] = model["normal_ramp_steps"]
        h5.attrs["shear_ramp_time"] = model["shear_ramp_time"]
        h5.attrs["shear_ramp_shape"] = model["shear_ramp_shape"]
        h5.attrs["normal_loading_mode"] = model["normal_loading_mode"]
        h5.attrs["shear_loading_mode"] = model["shear_loading_mode"]
        h5.attrs["shear_force_boundary"] = model["shear_force_boundary"]
        h5.attrs["shear_displacement_boundary"] = model["shear_displacement_boundary"]
        h5.attrs["normal_stress"] = model["normal_stress"]
        h5.attrs["normal_displacement"] = model["normal_displacement"]
        h5.attrs["normal_displacement_estimate"] = model["normal_displacement_estimate"]
        h5.attrs["shear_displacement_k"] = model["shear_displacement_k"]
        h5.attrs["shear_displacement_s"] = model["shear_displacement_s"]
        h5.attrs["quasistatic_shear_fraction"] = model[
            "quasistatic_shear_fraction"
        ]
        h5.attrs["quasistatic_shear_target"] = model[
            "quasistatic_shear_target"
        ]
        h5.attrs["quasistatic_shear_start_time"] = model[
            "quasistatic_shear_start_time"
        ]
        h5.attrs["quasistatic_shear_ramp_time"] = model[
            "quasistatic_shear_ramp_time"
        ]
        if model["normal_relaxation_time"] is not None:
            h5.attrs["normal_relaxation_time"] = model[
                "normal_relaxation_time"
            ]
            # Legacy alias retained for older post-processing scripts.
            h5.attrs["quasistatic_damping_time"] = model[
                "quasistatic_damping_time"
            ]
        h5.attrs["mu_s_start_fraction"] = model["mu_s_start_fraction"]
        h5.attrs["mu_s_end_fraction"] = model["mu_s_end_fraction"]
        h5.attrs["pw_length"] = model["pw_length"]
        h5.attrs["pw_mu_s_ratio"] = model["pw_mu_s_ratio"]
        h5.attrs["pw_transition_length"] = model["pw_transition_length"]
        h5.attrs["leading_edge_guard_length"] = model[
            "leading_edge_guard_length"
        ]
        h5.attrs["leading_edge_guard_mu_s_ratio"] = model[
            "leading_edge_guard_mu_s_ratio"
        ]
        h5.attrs["leading_edge_guard_transition_length"] = model[
            "leading_edge_guard_transition_length"
        ]
        h5.attrs["leading_edge_tangential_taper_length"] = model[
            "leading_edge_tangential_taper_length"
        ]
        h5.attrs["leading_edge_tangential_plateau_length"] = model[
            "leading_edge_tangential_plateau_length"
        ]
        h5.attrs["leading_edge_tangential_taper_ratio"] = model[
            "leading_edge_tangential_taper_ratio"
        ]
        h5.attrs["leading_edge_creep_length"] = model["leading_edge_creep_length"]
        h5.attrs["leading_edge_creep_transition_length"] = model[
            "leading_edge_creep_transition_length"
        ]
        h5.attrs["leading_edge_creep_mu"] = model["leading_edge_creep_mu"]
        h5.attrs["leading_edge_creep_mu_k"] = model["leading_edge_creep_mu_k"]
        h5.attrs["leading_edge_creep_relaxation_time"] = model[
            "leading_edge_creep_relaxation_time"
        ]
        h5.attrs["friction_law"] = model["friction_law"]
        h5.attrs["rsf_reference_friction"] = model["rsf_reference_friction"]
        h5.attrs["rsf_direct_effect"] = model["rsf_direct_effect"]
        h5.attrs["rsf_state_effect"] = model["rsf_state_effect"]
        h5.attrs["rsf_reference_velocity"] = model["rsf_reference_velocity"]
        h5.attrs["rsf_reference_state"] = model["rsf_reference_state"]
        h5.attrs["rsf_characteristic_slip"] = model[
            "rsf_characteristic_slip"
        ]
        h5.attrs["rsf_initial_state"] = model["rsf_initial_state"]
        if model["rsf_profile_spec"] is not None:
            h5.attrs["rsf_profile_spec_json"] = json.dumps(model["rsf_profile_spec"])
        if model["shear_loading_stiffness"] is not None:
            h5.attrs["shear_loading_stiffness"] = model[
                "shear_loading_stiffness"
            ]
        h5.attrs["stop_shear_loading_on_rupture"] = int(
            model["stop_shear_loading_on_rupture"]
        )
        h5.attrs["shear_loading_stop_slip"] = model["shear_loading_stop_slip"]
        if model["shear_loading_stop_velocity"] is not None:
            h5.attrs["shear_loading_stop_velocity"] = model[
                "shear_loading_stop_velocity"
            ]
        h5.attrs["shear_loading_stop_uses_critical_profile"] = int(
            model["shear_loading_stop_uses_critical_profile"]
        )
        if model["shear_loading_stop_min_y"] is not None:
            h5.attrs["shear_loading_stop_min_y"] = model[
                "shear_loading_stop_min_y"
            ]
        if model["shear_loading_stop_max_y"] is not None:
            h5.attrs["shear_loading_stop_max_y"] = model[
                "shear_loading_stop_max_y"
            ]
        h5.attrs["critical_slip"] = model["critical_slip"]
        h5.attrs["mu_k"] = model["mu_k"]
        h5.attrs["loading_edge_nucleation_length"] = model[
            "loading_edge_nucleation_length"
        ]
        h5.attrs["loading_edge_critical_slip"] = model[
            "loading_edge_critical_slip"
        ]
        h5.attrs["relax_tangential_contact_during_normal"] = int(
            model["relax_tangential_contact_during_normal"]
        )
        h5.attrs["lock_shear_edge_during_normal"] = int(
            config.lock_shear_edge_during_normal
        )
        h5.create_dataset("history_columns", data=np.asarray(history_columns, dtype="S"))
        h5.create_dataset("phase_id", shape=(total_frames,), dtype="i4")
        h5.create_dataset("step_id", shape=(total_frames,), dtype="i4")
        h5.create_dataset(
            "history",
            shape=(total_frames, len(history_columns)),
            dtype="f4",
            compression=compression,
            chunks=(min(256, total_frames), len(history_columns)),
        )
        interface_plot_mask = np.isin(
            np.asarray(master_nodes), np.asarray(interface_plot_master_nodes)
        )

        moving_grp = _create_group_datasets(
            h5, moving, "moving", total_frames, n_moving, int(moving.mesh.elements.shape[0])
        )
        stationary_grp = _create_group_datasets(
            h5,
            stationary,
            "stationary",
            total_frames,
            n_stationary,
            int(stationary.mesh.elements.shape[0]),
        )
        iface = h5.create_group("interface")
        iface.attrs["mu_static"] = friction.mu_s
        iface.attrs["mu_kinetic"] = friction.mu_k
        iface.attrs["critical_slip"] = model["critical_slip"]
        iface.attrs["loading_edge_nucleation_length"] = model[
            "loading_edge_nucleation_length"
        ]
        iface.attrs["loading_edge_critical_slip"] = model[
            "loading_edge_critical_slip"
        ]
        iface.attrs["mu_static_start_fraction"] = model["mu_s_start_fraction"]
        iface.attrs["mu_static_end_fraction"] = model["mu_s_end_fraction"]
        iface.attrs["pw_length"] = model["pw_length"]
        iface.attrs["pw_mu_s_ratio"] = model["pw_mu_s_ratio"]
        iface.attrs["pw_transition_length"] = model["pw_transition_length"]
        iface.attrs["leading_edge_guard_length"] = model[
            "leading_edge_guard_length"
        ]
        iface.attrs["leading_edge_guard_mu_s_ratio"] = model[
            "leading_edge_guard_mu_s_ratio"
        ]
        iface.attrs["leading_edge_guard_transition_length"] = model[
            "leading_edge_guard_transition_length"
        ]
        iface.attrs["leading_edge_tangential_taper_length"] = model[
            "leading_edge_tangential_taper_length"
        ]
        iface.attrs["leading_edge_tangential_plateau_length"] = model[
            "leading_edge_tangential_plateau_length"
        ]
        iface.attrs["leading_edge_tangential_taper_ratio"] = model[
            "leading_edge_tangential_taper_ratio"
        ]
        iface.attrs["leading_edge_creep_length"] = model["leading_edge_creep_length"]
        iface.attrs["leading_edge_creep_transition_length"] = model[
            "leading_edge_creep_transition_length"
        ]
        iface.attrs["leading_edge_creep_mu"] = model["leading_edge_creep_mu"]
        iface.attrs["leading_edge_creep_mu_k"] = model[
            "leading_edge_creep_mu_k"
        ]
        iface.attrs["leading_edge_creep_relaxation_time"] = model[
            "leading_edge_creep_relaxation_time"
        ]
        iface.attrs["friction_law"] = model["friction_law"]
        iface.attrs["rsf_reference_friction"] = model[
            "rsf_reference_friction"
        ]
        iface.attrs["rsf_direct_effect"] = model["rsf_direct_effect"]
        iface.attrs["rsf_state_effect"] = model["rsf_state_effect"]
        iface.attrs["rsf_reference_velocity"] = model[
            "rsf_reference_velocity"
        ]
        iface.attrs["rsf_reference_state"] = model["rsf_reference_state"]
        iface.attrs["rsf_characteristic_slip"] = model[
            "rsf_characteristic_slip"
        ]
        iface.attrs["rsf_initial_state"] = model["rsf_initial_state"]
        iface.create_dataset("master_nodes", data=np.asarray(interface_plot_master_nodes))
        iface.create_dataset(
            "slave_nodes",
            data=np.asarray(slave_nodes[interface_plot_mask], dtype=np.int32),
        )
        iface.create_dataset(
            "contact_line_y",
            data=np.asarray(
                moving.mesh.coords[np.asarray(interface_plot_master_nodes), 1],
                dtype=np.float32,
            ),
        )
        if dimension == 3:
            iface.create_dataset(
                "contact_line_z",
                data=np.asarray(
                    moving.mesh.coords[np.asarray(interface_plot_master_nodes), 2],
                    dtype=np.float32,
                ),
            )
        iface.create_dataset(
            "mu_static_profile",
            data=np.asarray(mu_s_profile[interface_plot_mask], dtype=np.float32),
        )
        iface.create_dataset(
            "mu_kinetic_profile",
            data=np.asarray(mu_k_profile[interface_plot_mask], dtype=np.float32),
        )
        iface.create_dataset(
            "critical_slip_profile",
            data=np.asarray(
                critical_slip_profile[interface_plot_mask], dtype=np.float32
            ),
        )
        iface.create_dataset(
            "creep_weight_profile",
            data=np.asarray(
                creep_weight_profile[interface_plot_mask], dtype=np.float32
            ),
        )
        iface.create_dataset(
            "tangential_penalty_profile",
            data=np.asarray(penalty_t[interface_plot_mask], dtype=np.float32),
        )
        if use_rate_state:
            for name in (
                "reference_friction",
                "direct_effect",
                "state_effect",
                "reference_velocity",
                "reference_state",
                "characteristic_slip",
                "initial_state",
            ):
                iface.create_dataset(
                    f"rsf_{name}_profile",
                    data=np.asarray(
                        rsf_parameters[name][interface_plot_mask], dtype=np.float32
                    ),
                )
        iface.create_dataset(
            "plastic_slip",
            shape=(total_frames, int(interface_plot_master_nodes.shape[0])),
            dtype="f4",
            compression=compression,
            chunks=(1, int(interface_plot_master_nodes.shape[0])),
        )
        iface.create_dataset(
            "cumulative_slip",
            shape=(total_frames, int(interface_plot_master_nodes.shape[0])),
            dtype="f4",
            compression=compression,
            chunks=(1, int(interface_plot_master_nodes.shape[0])),
        )
        iface.create_dataset(
            "rsf_state",
            shape=(total_frames, int(interface_plot_master_nodes.shape[0])),
            dtype="f4",
            compression=compression,
            chunks=(1, int(interface_plot_master_nodes.shape[0])),
        )
        iface.create_dataset(
            "friction_coefficient",
            shape=(total_frames, int(interface_plot_master_nodes.shape[0])),
            dtype="f4",
            compression=compression,
            chunks=(1, int(interface_plot_master_nodes.shape[0])),
        )
        iface.create_dataset(
            "friction_velocity",
            shape=(total_frames, int(interface_plot_master_nodes.shape[0])),
            dtype="f4",
            compression=compression,
            chunks=(1, int(interface_plot_master_nodes.shape[0])),
        )
        iface.create_dataset(
            "friction_strength",
            shape=(total_frames, int(interface_plot_master_nodes.shape[0])),
            dtype="f4",
            compression=compression,
            chunks=(1, int(interface_plot_master_nodes.shape[0])),
        )
        iface.create_dataset(
            "slip_rate",
            shape=(total_frames, int(interface_plot_master_nodes.shape[0])),
            dtype="f4",
            compression=compression,
            chunks=(1, int(interface_plot_master_nodes.shape[0])),
        )

        if separate_interface_output:
            high = h5.create_group("interface_high_rate")
            high.attrs["sampling"] = "independent high-rate interface history"
            high["contact_line_y"] = iface["contact_line_y"]
            high["master_nodes"] = iface["master_nodes"]
            high["slave_nodes"] = iface["slave_nodes"]
            high.create_dataset(
                "history",
                shape=(total_interface_frames, len(history_columns)),
                dtype="f4",
                compression=compression,
                chunks=(min(256, total_interface_frames), len(history_columns)),
            )
            high.create_dataset("history_columns", data=np.asarray(history_columns, dtype="S"))
            high.create_dataset("phase_id", shape=(total_interface_frames,), dtype="i4")
            high.create_dataset("step_id", shape=(total_interface_frames,), dtype="i4")
            for name in (
                "plastic_slip",
                "cumulative_slip",
                "rsf_state",
                "friction_coefficient",
                "friction_velocity",
                "friction_strength",
                "slip_rate",
            ):
                high.create_dataset(
                    name,
                    shape=(
                        total_interface_frames,
                        int(interface_plot_master_nodes.shape[0]),
                    ),
                    dtype="f4",
                    compression=compression,
                    chunks=(1, int(interface_plot_master_nodes.shape[0])),
                )

        if resume:
            for name in list(h5.keys()):
                if name != "_resume_stash":
                    del h5[name]
            stash = h5["_resume_stash"]
            for name in list(stash.keys()):
                h5.move(f"_resume_stash/{name}", name)
            del h5["_resume_stash"]
            if h5["history"].shape != (total_frames, len(history_columns)):
                raise ValueError("Existing HDF5 history shape does not match the case.")
            if separate_interface_output and h5[
                "interface_high_rate/history"
            ].shape != (total_interface_frames, len(history_columns)):
                raise ValueError(
                    "Existing high-rate interface shape does not match the case."
                )
            h5.attrs["resumed_from_checkpoint"] = 1
            h5.attrs["checkpoint_phase_id"] = resume_phase_id
            h5.attrs["checkpoint_step_id"] = resume_step_id
            h5.attrs["saved_frames"] = frame_count
            h5.attrs["saved_interface_frames"] = interface_frame_count
        elif include_initial_frame:
            save_frame(h5, 0, carry, initial_row, phase_id=0, step_id=0)
            frame_count = 1
            if separate_interface_output:
                save_high_rate_interface(
                    h5, 0, carry, initial_row, phase_id=0, step_id=0
                )
                interface_frame_count = 1
        else:
            frame_count = 0
            interface_frame_count = 0

        def save_checkpoint(
            phase_id: int,
            step_id: int,
            current_carry: tuple[jax.Array, ...],
        ) -> None:
            nonlocal last_checkpoint_time
            if checkpoint_path is None:
                return
            h5.flush()
            metadata = {
                "version": 1,
                "carry_count": len(current_carry),
                "dt": float(dt),
                "pressure_steps": int(model["pressure_steps"]),
                "shear_steps": int(model["shear_steps"]),
                "total_frames": total_frames,
                "total_interface_frames": total_interface_frames,
                "phase_id": int(phase_id),
                "step_id": int(step_id),
                "frame_count": int(frame_count),
                "interface_frame_count": int(interface_frame_count),
                "saved_utc": time.time(),
            }
            _write_simulation_checkpoint(checkpoint_path, current_carry, metadata)
            h5.attrs["checkpoint_phase_id"] = int(phase_id)
            h5.attrs["checkpoint_step_id"] = int(step_id)
            h5.attrs["checkpoint_frame_count"] = int(frame_count)
            h5.attrs["checkpoint_interface_frame_count"] = int(
                interface_frame_count
            )
            h5.attrs["checkpoint_path"] = str(checkpoint_path)
            h5.flush()
            last_checkpoint_time = time.monotonic()

        phases = [
            (
                1,
                "normal",
                np.column_stack(
                    [
                        np.asarray(model["normal_schedule_pressure"]),
                        np.asarray(model["pressure_schedule"]),
                        np.asarray(model["normal_displacement_pressure"]),
                        np.asarray(model["normal_velocity_pressure"]),
                        np.asarray(model["shear_displacement_pressure"]),
                        np.asarray(model["shear_velocity_pressure"]),
                    ]
                ),
                pressure_sample_stops,
                pressure_interface_stops,
                normal_phase_zero_dofs,
                prescribed_dofs,
            ),
            (
                2,
                "shear",
                np.column_stack(
                    [
                        np.asarray(model["normal_schedule_shear"]),
                        np.asarray(model["shear_schedule"]),
                        np.asarray(model["normal_displacement_shear"]),
                        np.asarray(model["normal_velocity_shear"]),
                        np.asarray(model["shear_displacement_shear"]),
                        np.asarray(model["shear_velocity_shear"]),
                    ]
                ),
                shear_sample_stops,
                shear_interface_stops,
                shear_phase_zero_dofs,
                prescribed_dofs,
            ),
        ]
        for (
            phase_id,
            phase_label,
            schedule,
            bulk_stops,
            interface_stops,
            zero_dofs,
            prescribed_dofs,
        ) in phases:
            if resume and phase_id < resume_phase_id:
                continue
            schedule_np = np.asarray(schedule)
            prev_stop = (
                resume_step_id
                if resume and phase_id == resume_phase_id
                else 0
            )
            simulation_stops = (
                np.union1d(bulk_stops, interface_stops)
                if separate_interface_output
                else bulk_stops
            )
            simulation_stops = simulation_stops[simulation_stops > prev_stop]
            bulk_stop_set = set(np.asarray(bulk_stops, dtype=np.int64).tolist())
            interface_stop_set = set(
                np.asarray(interface_stops, dtype=np.int64).tolist()
            )
            for stop in simulation_stops:
                stop = int(stop)
                chunk = jnp.asarray(schedule_np[prev_stop:stop], dtype=dtype)
                carry, outputs = advance_chunk(
                    carry,
                    chunk,
                    zero_dofs,
                    prescribed_dofs,
                    phase_label,
                )
                row = np.asarray(outputs[-1])
                if separate_interface_output and stop in interface_stop_set:
                    save_high_rate_interface(
                        h5,
                        interface_frame_count,
                        carry,
                        row,
                        phase_id=phase_id,
                        step_id=stop,
                    )
                    interface_frame_count += 1
                if stop in bulk_stop_set:
                    save_frame(
                        h5,
                        frame_count,
                        carry,
                        row,
                        phase_id=phase_id,
                        step_id=stop,
                    )
                    frame_count += 1
                prev_stop = stop
                now = time.monotonic()
                interval_due = (
                    checkpoint_interval_seconds is not None
                    and now - last_checkpoint_time >= checkpoint_interval_seconds
                )
                deadline_reached = (
                    checkpoint_deadline_monotonic is not None
                    and now >= checkpoint_deadline_monotonic
                )
                phase_complete = stop == int(simulation_stops[-1])
                if phase_complete and phase_id == 1 and normal_relaxation:
                    stored_energy = max(abs(float(row[7])) + abs(float(row[8])), 1.0e-30)
                    slip_candidates = _shear_loading_stop_candidates(
                        carry[3],
                        carry[5],
                        shear_loading_stop_mask,
                        critical_slip_profile,
                        shear_loading_stop_slip,
                        shear_loading_stop_uses_critical_profile,
                        None,
                    )
                    velocity_candidates = (
                        shear_loading_stop_mask
                        if shear_loading_stop_velocity is None
                        else shear_loading_stop_mask
                        & (jnp.abs(carry[5]) >= shear_loading_stop_velocity)
                    )
                    trigger_candidates = _shear_loading_stop_candidates(
                        carry[3],
                        carry[5],
                        shear_loading_stop_mask,
                        critical_slip_profile,
                        shear_loading_stop_slip,
                        shear_loading_stop_uses_critical_profile,
                        shear_loading_stop_velocity,
                    )
                    handoff_values = {
                        "kinetic_energy": float(row[9]),
                        "post_reset_kinetic_energy": 0.0,
                        "stored_energy": stored_energy,
                        "kinetic_ratio": float(row[9]) / stored_energy,
                        "max_slip": float(row[5]),
                        "max_slip_rate": float(jnp.max(jnp.abs(carry[5]))),
                        "stop_slip_threshold_exceeded": int(
                            jnp.any(slip_candidates)
                        ),
                        "stop_velocity_threshold_exceeded": int(
                            jnp.any(velocity_candidates)
                        ),
                        "stop_trigger_reached": int(jnp.any(trigger_candidates)),
                        "applied_displacement": float(row[10]),
                        "velocity_reset": 1,
                    }
                    prefixes = ["normal_relaxation_handoff"]
                    if quasistatic_preloading:
                        prefixes.append("quasistatic_handoff")
                    for prefix in prefixes:
                        for name, value in handoff_values.items():
                            h5.attrs[f"{prefix}_{name}"] = value
                    # A relaxed preload represents a static initial condition.
                    # Remove its tiny residual velocity before explicit shear starts.
                    carry = (
                        carry[0],
                        jnp.zeros_like(carry[1]),
                        *carry[2:],
                    )
                    h5.flush()
                if (
                    checkpoint_path is not None
                    and (
                        interval_due
                        or deadline_reached
                        or checkpoint_requested["value"]
                        or phase_complete
                    )
                ):
                    save_checkpoint(phase_id, stop, carry)
                if deadline_reached or checkpoint_requested["value"]:
                    restore_signal_handlers()
                    raise SimulationCheckpointed(
                        f"Checkpoint saved at phase {phase_id}, step {stop}: "
                        f"{checkpoint_path}"
                    )

        h5.attrs["saved_frames"] = frame_count
        h5.attrs["saved_interface_frames"] = (
            interface_frame_count if separate_interface_output else frame_count
        )

    if checkpoint_path is not None:
        checkpoint_path.unlink(missing_ok=True)
    restore_signal_handlers()

    (
        final_u,
        final_v,
        final_plastic,
        final_cum,
        final_rsf_state,
        final_slip_rate,
        final_friction_coefficient,
        final_friction_velocity,
        final_friction_strength,
        _final_time,
        _loading_stopped,
        _applied_shear_displacement,
    ) = carry
    final_time = dt * (model["pressure_steps"] + model["shear_steps"])
    with h5py.File(data_path, "r") as h5:
        history = np.asarray(h5["history"][:frame_count])
        stop_history = (
            np.asarray(h5["interface_high_rate/history"][:interface_frame_count])
            if separate_interface_output
            else history
        )
    stop_indices = np.flatnonzero(stop_history[:, 11] > 0.5)
    loading_stop_time = (
        float(stop_history[stop_indices[0], 0]) if stop_indices.size else None
    )
    loading_stop_time_in_shear = (
        loading_stop_time - model["pressure_time"]
        if loading_stop_time is not None
        else None
    )
    loading_stop_displacement = (
        float(stop_history[stop_indices[0], 10]) if stop_indices.size else None
    )
    def read_handoff(prefix: str) -> dict[str, float | bool]:
        handoff_names = (
            "kinetic_energy",
            "post_reset_kinetic_energy",
            "stored_energy",
            "kinetic_ratio",
            "max_slip",
            "max_slip_rate",
            "applied_displacement",
        )
        with h5py.File(data_path, "r") as h5:
            handoff = {
                name: float(h5.attrs[f"{prefix}_{name}"])
                for name in handoff_names
            }
            handoff["stop_slip_threshold_exceeded"] = bool(
                h5.attrs[f"{prefix}_stop_slip_threshold_exceeded"]
            )
            handoff["stop_velocity_threshold_exceeded"] = bool(
                h5.attrs[f"{prefix}_stop_velocity_threshold_exceeded"]
            )
            handoff["stop_trigger_reached"] = bool(
                h5.attrs[f"{prefix}_stop_trigger_reached"]
            )
            handoff["velocity_reset"] = bool(
                h5.attrs[f"{prefix}_velocity_reset"]
            )
        return handoff

    normal_relaxation_handoff = (
        read_handoff("normal_relaxation_handoff")
        if normal_relaxation
        else None
    )
    quasistatic_handoff = (
        read_handoff("quasistatic_handoff")
        if quasistatic_preloading
        else None
    )

    summary = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "dtype": str(history.dtype),
        "dimension": dimension,
        "thickness": float(model["thickness"]),
        "mesh_size": config.mesh_size,
        "cfl": model["cfl"],
        "contact_safety_factor": model["contact_safety_factor"],
        "time_step_override": model["time_step_override"],
        "dt": dt,
        "dt_stable_limit": model["dt_stable_limit"],
        "dt_bulk": model["dt_bulk"],
        "dt_contact": model["dt_contact"],
        "dt_stability_limiter": model["dt_stability_limiter"],
        "dt_limiter": model["dt_limiter"],
        "pressure_time": model["pressure_time"],
        "shear_time": model["shear_time"],
        "normal_ramp_time": model["normal_ramp_time"],
        "normal_ramp_steps": model["normal_ramp_steps"],
        "shear_ramp_time": model["shear_ramp_time"],
        "shear_ramp_shape": model["shear_ramp_shape"],
        "stop_shear_loading_on_rupture": model[
            "stop_shear_loading_on_rupture"
        ],
        "shear_loading_stop_slip": model["shear_loading_stop_slip"],
        "shear_loading_stop_velocity": model["shear_loading_stop_velocity"],
        "shear_loading_stop_uses_critical_profile": model[
            "shear_loading_stop_uses_critical_profile"
        ],
        "shear_loading_stop_min_y": model["shear_loading_stop_min_y"],
        "shear_loading_stop_max_y": model["shear_loading_stop_max_y"],
        "shear_loading_stop_time": loading_stop_time,
        "shear_loading_stop_time_in_shear": loading_stop_time_in_shear,
        "shear_loading_stop_displacement": loading_stop_displacement,
        "relax_tangential_contact_during_normal": model[
            "relax_tangential_contact_during_normal"
        ],
        "pressure_steps": model["pressure_steps"],
        "shear_steps": model["shear_steps"],
        "pressure_frames_target": int(frames_per_phase),
        "shear_frames_target": int(shear_frames_per_phase),
        "pressure_frames_saved": int(n_press_frames),
        "shear_frames_saved": int(n_shear_frames),
        "interface_pressure_frames_target": int(interface_frames_per_phase),
        "interface_shear_frames_target": int(shear_interface_frames_per_phase),
        "interface_pressure_frames_saved": int(n_press_interface_frames),
        "interface_shear_frames_saved": int(n_shear_interface_frames),
        "tau_k": model["tau_k"],
        "tau_s": model["tau_s"],
        "shear_scale": model["shear_scale"],
        "normal_stress": model["normal_stress"],
        "normal_loading_mode": normal_loading_mode,
        "shear_loading_mode": shear_loading_mode,
        "shear_loading_stiffness": model["shear_loading_stiffness"],
        "shear_force_boundary": model["shear_force_boundary"],
        "shear_displacement_boundary": model["shear_displacement_boundary"],
        "normal_displacement": model["normal_displacement"],
        "normal_displacement_estimate": model["normal_displacement_estimate"],
        "shear_displacement_k": model["shear_displacement_k"],
        "shear_displacement_s": model["shear_displacement_s"],
        "quasistatic_shear_fraction": model["quasistatic_shear_fraction"],
        "quasistatic_shear_target": model["quasistatic_shear_target"],
        "quasistatic_shear_start_time": model[
            "quasistatic_shear_start_time"
        ],
        "quasistatic_shear_ramp_time": model["quasistatic_shear_ramp_time"],
        "normal_relaxation_time": model["normal_relaxation_time"],
        "normal_relaxation_handoff": normal_relaxation_handoff,
        "quasistatic_damping_time": model["quasistatic_damping_time"],
        "quasistatic_handoff": quasistatic_handoff,
        "critical_slip": model["critical_slip"],
        "mu_k": model["mu_k"],
        "loading_edge_nucleation_length": model[
            "loading_edge_nucleation_length"
        ],
        "loading_edge_critical_slip": model["loading_edge_critical_slip"],
        "mu_s_start_fraction": model["mu_s_start_fraction"],
        "mu_s_end_fraction": model["mu_s_end_fraction"],
        "pw_length": model["pw_length"],
        "pw_mu_s_ratio": model["pw_mu_s_ratio"],
        "pw_transition_length": model["pw_transition_length"],
        "leading_edge_guard_length": model["leading_edge_guard_length"],
        "leading_edge_guard_mu_s_ratio": model[
            "leading_edge_guard_mu_s_ratio"
        ],
        "leading_edge_guard_transition_length": model[
            "leading_edge_guard_transition_length"
        ],
        "leading_edge_tangential_taper_length": model[
            "leading_edge_tangential_taper_length"
        ],
        "leading_edge_tangential_plateau_length": model[
            "leading_edge_tangential_plateau_length"
        ],
        "leading_edge_tangential_taper_ratio": model[
            "leading_edge_tangential_taper_ratio"
        ],
        "friction_law": model["friction_law"],
        "rsf_reference_friction": model["rsf_reference_friction"],
        "rsf_direct_effect": model["rsf_direct_effect"],
        "rsf_state_effect": model["rsf_state_effect"],
        "rsf_reference_velocity": model["rsf_reference_velocity"],
        "rsf_reference_state": model["rsf_reference_state"],
        "rsf_characteristic_slip": model["rsf_characteristic_slip"],
        "rsf_initial_state": model["rsf_initial_state"],
        "rsf_profile": model["rsf_profile_spec"],
        "leading_edge_creep_length": model["leading_edge_creep_length"],
        "leading_edge_creep_transition_length": model[
            "leading_edge_creep_transition_length"
        ],
        "leading_edge_creep_mu": model["leading_edge_creep_mu"],
        "leading_edge_creep_mu_k": model["leading_edge_creep_mu_k"],
        "leading_edge_creep_relaxation_time": model[
            "leading_edge_creep_relaxation_time"
        ],
        "normal_penalty": float(penalty_n),
        "tangential_penalty": model["tangential_penalty"],
        "lock_shear_edge_during_normal": bool(config.lock_shear_edge_during_normal),
        "moving_nodes": n_moving,
        "stationary_nodes": n_stationary,
        "moving_elements": int(moving.mesh.elements.shape[0]),
        "stationary_elements": int(stationary.mesh.elements.shape[0]),
        "final_time": float(final_time),
        "final_max_slip": float(jnp.max(final_cum)),
        "final_max_penetration": float(history[-1, 4]),
        "final_avg_tau": float(history[-1, 2]),
        "final_avg_sigma_n": float(history[-1, 3]),
        "saved_frames": int(frame_count),
        "data_path": str(data_path),
        "legacy": {
            "materials": {name: asdict(mat) for name, mat in case.materials.items()},
            "friction": asdict(case.friction),
            "effective_friction": asdict(model["friction"]),
            "simulation": asdict(case.simulation),
            "geometry": {
                "moving": asdict(case.moving),
                "stationary": asdict(case.stationary),
            },
        },
    }

    return {
        "summary": summary,
        "history": history,
        "columns": history_columns,
        "final_u": np.asarray(final_u),
        "final_v_half": np.asarray(final_v),
        "final_plastic_slip": np.asarray(final_plastic),
        "final_cum_slip": np.asarray(final_cum),
        "final_rsf_state": np.asarray(final_rsf_state),
        "final_slip_rate": np.asarray(final_slip_rate),
        "final_friction_coefficient": np.asarray(final_friction_coefficient),
        "final_friction_velocity": np.asarray(final_friction_velocity),
        "final_friction_strength": np.asarray(final_friction_strength),
    }


def save_history_plots(
    result: dict[str, Any],
    plot_dir: Path,
    *,
    prefix: str = "velocity_weakening_tatva",
    extension: str = ".pdf",
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plot_dir.mkdir(parents=True, exist_ok=True)
    history = np.asarray(result["history"])
    columns = list(result["columns"])
    col = {name: idx for idx, name in enumerate(columns)}
    time_ms = history[:, col["time"]] * 1e3
    stop_time = result.get("summary", {}).get("shear_loading_stop_time")
    stop_time_ms = None if stop_time is None else float(stop_time) * 1e3

    def mark_loading_stop(axis: Any, *, label: bool = False) -> None:
        if stop_time_ms is not None:
            axis.axvline(
                stop_time_ms,
                color="black",
                ls=":",
                lw=1.5,
                label="Loading stopped" if label else None,
            )

    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if result["summary"]["shear_loading_mode"] in {
        "displacement",
        "spring-displacement",
    }:
        ax.plot(
            time_ms,
            history[:, col["applied_shear_displacement"]],
            label="Actuator shear displacement",
            lw=2,
        )
        ax.set_ylabel("Shear displacement")
    else:
        ax.plot(
            time_ms,
            history[:, col["applied_shear"]],
            label="Applied shear",
            lw=2,
        )
        ax.set_ylabel("Applied shear")
    mark_loading_stop(ax, label=True)
    ax.set_xlabel("Time [ms]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(
        time_ms,
        history[:, col["avg_tau"]],
        label="Average interface shear",
        lw=2,
        color="tab:orange",
    )
    ax2.plot(
        time_ms,
        history[:, col["avg_sigma_n"]],
        label="Average interface normal",
        lw=2,
        color="tab:green",
    )
    ax2.set_ylabel("Interface traction")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper center")
    path = plot_dir / f"{prefix}_tractions{extension}"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7.0), sharex=True)
    axes[0].plot(time_ms, history[:, col["max_slip"]], label="Max cumulative slip", lw=2)
    axes[0].plot(time_ms, history[:, col["max_penetration"]], label="Max penetration", lw=2)
    axes[0].set_ylabel("Slip / penetration")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    mark_loading_stop(axes[0])
    axes[1].plot(
        time_ms,
        history[:, col["mu_eff_mean"]],
        label="Mean effective friction",
        lw=2,
        color="tab:green",
    )
    axes[1].set_xlabel("Time [ms]")
    axes[1].set_ylabel("Effective friction")
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    mark_loading_stop(axes[1])
    path = plot_dir / f"{prefix}_interface_state{extension}"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(time_ms, history[:, col["elastic_energy"]], label="Elastic energy", lw=2)
    ax.plot(time_ms, history[:, col["interface_energy"]], label="Interface energy", lw=2)
    ax.plot(time_ms, history[:, col["kinetic_energy"]], label="Kinetic energy", lw=2)
    ax.plot(
        time_ms,
        history[:, col["elastic_energy"]]
        + history[:, col["interface_energy"]]
        + history[:, col["kinetic_energy"]],
        label="Total tracked energy",
        lw=2,
        ls="--",
    )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3)
    mark_loading_stop(ax, label=True)
    ax.legend()
    path = plot_dir / f"{prefix}_energies{extension}"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    return saved
