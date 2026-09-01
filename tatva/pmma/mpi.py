"""MPI helpers for replicated-state PMMA explicit dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from tatva import Mesh, Operator


@dataclass(frozen=True)
class MPIContext:
    """Small optional-MPI facade that keeps serial imports dependency-free."""

    comm: Any | None
    rank: int = 0
    size: int = 1

    @property
    def enabled(self) -> bool:
        return self.comm is not None and self.size > 1

    @property
    def is_root(self) -> bool:
        return self.rank == 0


def get_mpi_context() -> MPIContext:
    """Return ``COMM_WORLD`` when launched with MPI, otherwise serial context."""
    try:
        from mpi4py import MPI
    except ImportError:
        return MPIContext(comm=None)

    comm = MPI.COMM_WORLD
    return MPIContext(comm=comm, rank=comm.Get_rank(), size=comm.Get_size())


def partition_operator(operator: Operator, rank: int, size: int) -> Operator:
    """Partition contiguous elements while retaining replicated global nodes."""
    if size <= 0:
        raise ValueError("MPI size must be positive.")
    if rank < 0 or rank >= size:
        raise ValueError(f"MPI rank {rank} is outside [0, {size}).")

    n_elements = int(operator.mesh.elements.shape[0])
    if size > n_elements:
        raise ValueError(
            f"Cannot partition {n_elements} elements across {size} MPI ranks."
        )
    start = rank * n_elements // size
    stop = (rank + 1) * n_elements // size
    local_elements = jnp.asarray(operator.mesh.elements[start:stop])
    local_mesh = Mesh(coords=operator.mesh.coords, elements=local_elements)
    batch_size = min(int(operator.batch_size), stop - start)
    return Operator(
        local_mesh,
        operator.element,
        batch_size=batch_size,
        cache_weights=operator.cache_weights,
    )


def make_allreduced_value_and_grad(
    local_energy: Callable[[jax.Array], jax.Array],
    context: MPIContext,
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    """Sum rank-local scalar energy and dense force in one MPI collective."""
    local_value_and_grad = jax.value_and_grad(local_energy)
    if not context.enabled:
        return jax.jit(local_value_and_grad)

    import mpi4jax
    from mpi4py import MPI

    comm = context.comm

    @jax.jit
    def allreduced(value: jax.Array) -> tuple[jax.Array, jax.Array]:
        local_value, local_grad = local_value_and_grad(value)
        packed = jnp.concatenate((local_value.reshape(1), local_grad.reshape(-1)))
        reduced = mpi4jax.allreduce(packed, op=MPI.SUM, comm=comm)
        return reduced[0], reduced[1:].reshape(local_grad.shape)

    return allreduced


def partition_counts(n_elements: int, size: int) -> np.ndarray:
    """Expose deterministic partition sizes for preflight and tests."""
    if n_elements < size or size <= 0:
        raise ValueError("Each MPI rank must own at least one element.")
    boundaries = np.arange(size + 1, dtype=np.int64) * n_elements // size
    return np.diff(boundaries)


def synchronize_flags(
    context: MPIContext, *flags: bool
) -> tuple[bool, ...]:
    """Make rank-local control-flow requests identical before the next collective."""
    if not context.enabled:
        return tuple(bool(flag) for flag in flags)

    from mpi4py import MPI

    local = np.asarray(flags, dtype=np.int8)
    global_flags = np.empty_like(local)
    context.comm.Allreduce(local, global_flags, op=MPI.MAX)
    return tuple(bool(value) for value in global_flags)
