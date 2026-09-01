import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tatva import Mesh, Operator, element
from tatva.pmma.mpi import (
    get_mpi_context,
    make_allreduced_value_and_grad,
    partition_counts,
    partition_operator,
    synchronize_flags,
)


def _energy(operator: Operator, displacement: jax.Array) -> jax.Array:
    gradient = operator.grad(displacement)
    return operator.integrate(jnp.sum(gradient * gradient, axis=(-2, -1)))


def test_partition_counts_are_balanced_and_complete():
    counts = partition_counts(23, 5)

    assert counts.sum() == 23
    assert counts.max() - counts.min() <= 1


def test_serial_control_flags_are_unchanged():
    assert synchronize_flags(get_mpi_context(), True, False) == (True, False)


def test_partitioned_element_energy_and_gradient_match_serial():
    mesh = Mesh.unit_square(8, 5)
    operator = Operator(mesh, element.Tri3())
    displacement = jnp.arange(mesh.coords.shape[0] * 2, dtype=jnp.float32).reshape(
        -1, 2
    )
    serial_energy, serial_gradient = jax.value_and_grad(
        lambda value: _energy(operator, value)
    )(displacement)

    local_results = [
        jax.value_and_grad(
            lambda value, local=partition_operator(operator, rank, 4): _energy(
                local, value
            )
        )(displacement)
        for rank in range(4)
    ]
    partitioned_energy = sum(result[0] for result in local_results)
    partitioned_gradient = sum(result[1] for result in local_results)

    np.testing.assert_allclose(partitioned_energy, serial_energy, rtol=1.0e-6)
    np.testing.assert_allclose(
        partitioned_gradient, serial_gradient, rtol=1.0e-6, atol=5.0e-6
    )


def test_mpi_allreduced_energy_and_gradient_match_serial():
    context = get_mpi_context()
    if not context.enabled:
        pytest.skip("Run with at least two MPI ranks.")

    mesh = Mesh.unit_square(8, 5)
    operator = Operator(mesh, element.Tri3())
    local_operator = partition_operator(operator, context.rank, context.size)
    displacement = jnp.arange(mesh.coords.shape[0] * 2, dtype=jnp.float32).reshape(
        -1, 2
    )
    distributed = make_allreduced_value_and_grad(
        lambda value: _energy(local_operator, value), context
    )

    distributed_energy, distributed_gradient = distributed(displacement)
    serial_energy, serial_gradient = jax.value_and_grad(
        lambda value: _energy(operator, value)
    )(displacement)

    np.testing.assert_allclose(distributed_energy, serial_energy, rtol=1.0e-6)
    np.testing.assert_allclose(
        distributed_gradient, serial_gradient, rtol=1.0e-6, atol=5.0e-6
    )
    assert synchronize_flags(context, context.rank == 0, False) == (True, False)
