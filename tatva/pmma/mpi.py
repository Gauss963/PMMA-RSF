"""Replicated-state MPI elastic assembly for the PMMA explicit solver.

Only element integration is partitioned; contact and constraints run identically
on every rank. The output is written by rank zero. Host control collectives use
a different communicator from JAX collectives to avoid asynchronous ordering.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tatva import Mesh, Operator


@dataclass(frozen=True)
class MPIContext:
    comm: Any = None
    compute_comm: Any = None
    rank: int = 0
    size: int = 1

    @property
    def enabled(self):
        return self.size > 1

    @property
    def is_root(self):
        return self.rank == 0


@lru_cache(maxsize=1)
def get_mpi_context():
    try:
        from mpi4py import MPI
    except ImportError:
        import os
        if any(int(os.getenv(key, "1")) > 1 for key in
               ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE")):
            raise RuntimeError("MPI launch requires mpi4py and mpi4jax.")
        return MPIContext()
    comm = MPI.COMM_WORLD
    return MPIContext(comm, comm.Dup(), comm.rank, comm.size)


def partition_operator(operator, context):
    if not context.enabled:
        return operator
    count = len(operator.mesh.elements)
    if count < context.size:
        raise ValueError("Each MPI rank needs at least one element per block.")
    first = context.rank * count // context.size
    last = (context.rank + 1) * count // context.size
    mesh = Mesh(coords=operator.mesh.coords, elements=operator.mesh.elements[first:last])
    batch = operator.batch_size
    return Operator(mesh, operator.element,
                    batch_size=None if batch is None else min(batch, last - first),
                    cache_weights=operator.cache_weights)


def make_allreduced_value_and_grad(energy, context):
    """Reduce both energy and its gradient, including additive extension energy."""
    local = jax.value_and_grad(energy, has_aux=True)
    if not context.enabled:
        return jax.jit(local)
    import mpi4jax
    from mpi4py import MPI

    @jax.jit
    def assembled(displacement):
        (value, auxiliary), gradient = local(displacement)
        packed = jnp.concatenate((jnp.stack((value, auxiliary)), gradient.ravel()))
        reduced = mpi4jax.allreduce(packed, op=MPI.SUM, comm=context.compute_comm)
        return (reduced[0], reduced[1]), reduced[2:].reshape(gradient.shape)

    return assembled


def synchronize_flags(context, *flags):
    if not context.enabled:
        return tuple(bool(flag) for flag in flags)
    from mpi4py import MPI
    send = np.asarray(flags, dtype=np.int8)
    received = np.empty_like(send)
    context.comm.Allreduce(send, received, op=MPI.MAX)
    return tuple(bool(flag) for flag in received)
