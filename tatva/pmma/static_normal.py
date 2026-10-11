"""Static equilibrium of the normal-loading phase for the two-block contact model.

The explicit normal ramp leaves undamped standing waves (local sigma_n swinging by
several MPa long after the ramp). Solving the end-of-normal-loading state as a
static equilibrium instead gives the shear phase a clean initial condition.

The equations are exactly those of ``run_simulation_dumped``: plane-strain linear
elasticity of both blocks, node-to-node penalty contact (normal kn * overlap,
tangential kt * (relative tangential displacement - plastic slip)), the full normal
traction, Dirichlet supports, and the shear-loading face either locked (uy = 0) or
held by the rigid-platen spring with the actuator at zero. Coulomb sliding where the
stuck solution exceeds mu_s * sigma_n (in practice the contact ends) and opening
contact are resolved by an active-set iteration; the static strength is the
undamaged mu_s profile.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from tatva import Mesh, sparse
from tatva_coloring import distance2_color_and_seeds


def _block_stiffness(block, material) -> sps.csr_matrix:
    """Plane-strain stiffness of one block, assembled from the solver's energy density."""
    mu, lmbda = float(material.mu), float(material.lmbda)
    operator = block.operator

    def energy(u_flat):
        grad_u = operator.grad(u_flat.reshape(-1, 2))
        eps = 0.5 * (grad_u + jnp.swapaxes(grad_u, -1, -2))
        density = mu * jnp.einsum("...ij,...ij->...", eps, eps) + 0.5 * lmbda * jnp.trace(
            eps, axis1=-2, axis2=-1
        ) ** 2
        return operator.integrate(density)

    n_dofs = 2 * block.n_nodes
    with jax.enable_x64(True):
        mesh = Mesh(coords=jnp.asarray(block.mesh.coords, dtype=jnp.float64),
                    elements=jnp.asarray(block.mesh.elements, dtype=jnp.int32))
        pattern = sparse.pattern_from_mesh(mesh, n_dofs_per_node=2)
        colors, _ = distance2_color_and_seeds(row_ptr=pattern.indptr, col_idx=pattern.indices,
                                              n_dofs=n_dofs)
        colored = sparse.ColoredMatrix.from_csr(pattern, colors=colors)
        hessian = sparse.jacfwd(jax.grad(energy), colored, color_batch_size=8)
        matrix = hessian(jnp.zeros(n_dofs, dtype=jnp.float64)).to_csr()
    return sps.csr_matrix(matrix)


def solve_static_normal_phase(model: dict[str, Any], config, *, max_iterations: int = 100,
                              actuator_displacement: float = 0.0,
                              plastic_slip: np.ndarray | None = None) -> dict[str, Any]:
    """Return the static end-of-normal-loading state as float64 arrays.

    ``actuator_displacement`` (rigid-platen spring only) adds a static shear preload,
    used for diagnostics such as where the stuck fault first reaches mu_s.
    ``plastic_slip`` gives stuck pairs a stress-free tangential offset (restart from a
    previous event on a healed fault); sliding pairs replace it by the Coulomb value.
    """
    if int(config.dimension) != 2:
        raise NotImplementedError("The static normal phase is implemented for 2-D models.")
    if model["normal_loading_mode"] != "stress":
        raise NotImplementedError("The static normal phase requires normal_loading_mode='stress'.")
    if model["shear_loading_mode"] == "spring-displacement" and not model["shear_spring_rigid_face"]:
        if not config.lock_shear_edge_during_normal:
            raise NotImplementedError(
                "A static normal phase with a spring needs shear_spring_rigid_face=true "
                "(or a locked shear edge)."
            )
    moving, stationary = model["moving"], model["stationary"]
    offset = 2 * moving.n_nodes
    total = offset + 2 * stationary.n_nodes
    stiffness = sps.block_diag(
        [_block_stiffness(moving, model["moving_material"]),
         _block_stiffness(stationary, model["stationary_material"])],
        format="csr",
    )

    master = np.asarray(model["master_nodes"], dtype=np.int64)
    slave = np.asarray(model["slave_nodes"], dtype=np.int64)
    weights = np.asarray(model["interface_weights"], dtype=np.float64)
    kn = float(model["penalty_n"])
    kt = np.asarray(model["penalty_t"], dtype=np.float64)
    # Static strength of the (healed) fault: mu_k + h (mu_s - mu_k).
    healing = float(model.get("healing_fraction", 1.0))
    mu_k = np.asarray(model["mu_k_profile"], dtype=np.float64)
    mu_s = mu_k + healing * (np.asarray(model["mu_s_profile"], dtype=np.float64) - mu_k)
    mx, my = 2 * master, 2 * master + 1
    sx, sy = offset + 2 * slave, offset + 2 * slave + 1
    force = np.asarray(model["force_normal"], dtype=np.float64)
    gap = np.asarray(model["interface_initial_gap"], dtype=np.float64)
    stuck_offset = np.zeros(master.size) if plastic_slip is None else np.asarray(plastic_slip, dtype=np.float64)

    # Degrees of freedom: Dirichlet supports, plus the shear face (locked, or one rigid platen).
    fixed = set(np.asarray(model["fixed_dofs"], dtype=np.int64).tolist())
    face = np.asarray(model["moving_shear_loading_dofs"], dtype=np.int64)
    rigid_face = (model["shear_loading_mode"] == "spring-displacement"
                  and model["shear_spring_rigid_face"] and not config.lock_shear_edge_during_normal)
    if config.lock_shear_edge_during_normal:
        fixed.update(np.asarray(model["moving_shear_edge_dofs"], dtype=np.int64).tolist())
    free = np.setdiff1d(np.arange(total), np.fromiter(fixed, dtype=np.int64))
    # Map full dofs -> reduced columns; every face dof of a rigid platen shares one column.
    column = np.full(total, -1, dtype=np.int64)
    column[free] = np.arange(free.size)
    if rigid_face:
        column[face] = column[face[0]]
        used = np.unique(column[column >= 0])
        remap = np.full(column.max() + 1, -1, dtype=np.int64)
        remap[used] = np.arange(used.size)
        column[column >= 0] = remap[column[column >= 0]]
    n_reduced = int(column.max()) + 1
    rows = np.flatnonzero(column >= 0)
    transform = sps.csr_matrix((np.ones(rows.size), (rows, column[rows])), shape=(total, n_reduced))

    reduced_elastic = (transform.T @ stiffness @ transform).tocsr()
    spring = 0.0
    if rigid_face:
        spring = float(model["shear_loading_stiffness"]) * float(np.sum(model["force_shear_unit"]))
    in_contact = gap <= 0.0
    sliding = np.zeros(master.size, dtype=bool)
    direction = np.zeros(master.size)
    history = []
    for iteration in range(1, max_iterations + 1):
        # Contact stiffness for the current state (internal force = K_c u).
        r, c, v = [], [], []
        def add(rows_, cols_, vals):
            r.append(rows_); c.append(cols_); v.append(vals)
        normal = weights * kn * in_contact
        for a, b, s in ((mx, mx, 1), (mx, sx, -1), (sx, mx, -1), (sx, sx, 1)):
            add(a, b, s * normal)
        stick = in_contact & ~sliding
        tangential = weights * kt * stick
        for a, b, s in ((my, my, 1), (my, sy, -1), (sy, my, -1), (sy, sy, 1)):
            add(a, b, s * tangential)
        # Sliding: tau = direction * mu_s * kn * overlap, a non-symmetric coupling.
        slide = weights * mu_s * kn * direction * (in_contact & sliding)
        for a, b, s in ((my, mx, 1), (my, sx, -1), (sy, mx, -1), (sy, sx, 1)):
            add(a, b, s * slide)
        contact = sps.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))),
                                 shape=(total, total))
        matrix = (reduced_elastic + transform.T @ contact @ transform).tocsc()
        if rigid_face:
            platen = int(column[face[0]])
            matrix = matrix + sps.csc_matrix(([spring], ([platen], [platen])), shape=matrix.shape)
        # Overlap = u_mx - u_sx - gap: the gap of active pairs moves to the right-hand side.
        rhs = force.copy()
        if rigid_face and actuator_displacement:
            # Spring force k (u_a - u_face): the u_a part is an external load on the platen.
            rhs[face] += spring * actuator_displacement / face.size
        np.add.at(rhs, mx, normal * gap)
        np.add.at(rhs, sx, -normal * gap)
        np.add.at(rhs, my, slide * gap)
        np.add.at(rhs, sy, -slide * gap)
        np.add.at(rhs, my, tangential * stuck_offset)   # stuck: tau = kt (tangent - stuck_offset)
        np.add.at(rhs, sy, -tangential * stuck_offset)
        q = spla.spsolve(matrix, transform.T @ rhs)
        u = transform @ q
        overlap = u[mx] - u[sx] - gap
        tangent = u[my] - u[sy]
        sigma = kn * overlap
        trial = kt * (tangent - stuck_offset)
        new_contact = overlap > 0.0
        new_sliding = new_contact & (np.abs(trial) > mu_s * sigma)
        new_direction = np.where(new_sliding, np.sign(trial), 0.0)
        history.append({"iteration": iteration, "open": int((~new_contact).sum()),
                        "sliding": int(new_sliding.sum())})
        if (np.array_equal(new_contact, in_contact) and np.array_equal(new_sliding, sliding)
                and np.array_equal(new_direction, direction)):
            break
        in_contact, sliding, direction = new_contact, new_sliding, new_direction
    else:
        raise RuntimeError(f"Static normal phase did not converge in {max_iterations} iterations: {history[-3:]}")

    sigma = np.where(in_contact, sigma, 0.0)  # open pairs carry no traction
    tau = np.where(sliding, direction * mu_s * sigma, kt * (tangent - stuck_offset)) * in_contact
    plastic = np.where(sliding, tangent - tau / kt, stuck_offset)
    residual = stiffness @ u + contact @ u - rhs
    if rigid_face:
        residual[face] += spring * u[face[0]] / face.size
    free_residual = transform.T @ residual
    ratio = np.where(sigma > 0.0, np.abs(tau) / np.where(sigma > 0.0, sigma, 1.0), 0.0)
    return {
        "u": u,
        "plastic_slip": plastic,
        "cumulative_slip": np.abs(plastic),
        "sigma_n": sigma,
        "tau": tau,
        "tau_over_sigma": ratio,
        "sliding_nodes": int(sliding.sum()),
        "open_nodes": int((~in_contact).sum()),
        "iterations": len(history),
        "relative_residual": float(np.linalg.norm(free_residual) / max(np.linalg.norm(transform.T @ force), 1e-30)),
        "loading_face_displacement": float(np.mean(u[face])),
    }
