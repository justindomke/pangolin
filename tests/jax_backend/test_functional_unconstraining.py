import math

import jax
import jax.numpy as jnp

from pangolin import functional, ir
from pangolin.jax_backend import lower, require_array


def compiled_simplex_bijector(transform: functional.Transform) -> ir.Bijector:
    """Compile a functional simplex Transform around a Dirichlet Op."""
    concentration = ir.RV(ir.Constant([1.0, 2.0, 3.0]))

    def dirichlet(alpha: ir.RV) -> ir.RV:
        return ir.RV(ir.Dirichlet(), alpha)

    transformed = transform(dirichlet)(concentration)
    assert isinstance(transformed.op, ir.Bijected)
    return transformed.op.bijector


def simplex_chart_log_det(bijector: ir.Bijector, x: jax.Array) -> jax.Array:
    """Numerically differentiate a simplex transform in first-K-minus-one coordinates."""

    def forward_from_chart(head: jax.Array) -> jax.Array:
        simplex = jnp.concatenate([head, jnp.array([1 - jnp.sum(head)])])
        return require_array(lower(bijector.forward, (simplex,)))

    jacobian = jax.jacfwd(forward_from_chart)(x[:-1])
    return jnp.linalg.slogdet(jacobian).logabsdet


def test_functional_dirichlet_transform_round_trips_and_has_expected_jacobian():
    concentration = ir.RV(ir.Constant([1.0, 2.0, 3.0]))
    value = ir.RV(ir.Dirichlet(), concentration)

    reconstructed = functional.unconstrain(value, ())[0]
    free = reconstructed.parents[0]
    assert isinstance(free.op, ir.Bijected)
    bijector = free.op.bijector

    x = jnp.array([0.2, 0.3, 0.5])
    y = require_array(lower(bijector.forward, (x,)))
    round_trip = require_array(lower(bijector.inverse, (y,)))
    log_det = require_array(lower(bijector.log_det_jac, (x, y)))

    expected_y = jnp.array(
        [
            (jnp.log(x[0]) - jnp.log(x[1])) / math.sqrt(2),
            (jnp.log(x[0]) + jnp.log(x[1]) - 2 * jnp.log(x[2])) / math.sqrt(6),
        ]
    )
    expected_log_det = -jnp.sum(jnp.log(x)) - 0.5 * jnp.log(3)

    assert y.shape == (2,)
    assert jnp.allclose(y, expected_y)
    assert jnp.allclose(round_trip, x)
    assert jnp.allclose(log_det, expected_log_det)


def test_functional_stick_breaking_round_trips_and_has_expected_jacobian():
    bijector = compiled_simplex_bijector(functional.tforms.stick_breaking)
    x = jnp.array([0.2, 0.3, 0.5])

    y = require_array(lower(bijector.forward, (x,)))
    round_trip = require_array(lower(bijector.inverse, (y,)))
    log_det = require_array(lower(bijector.log_det_jac, (x, y)))

    assert jnp.allclose(y, jnp.array([jnp.log(0.2 / 0.8), jnp.log(0.375 / 0.625)]))
    assert jnp.allclose(round_trip, x)
    assert jnp.allclose(log_det, simplex_chart_log_det(bijector, x))


def test_functional_softmax_centered_round_trips_and_has_expected_jacobian():
    bijector = compiled_simplex_bijector(functional.tforms.softmax_centered)
    x = jnp.array([0.2, 0.3, 0.5])

    y = require_array(lower(bijector.forward, (x,)))
    round_trip = require_array(lower(bijector.inverse, (y,)))
    log_det = require_array(lower(bijector.log_det_jac, (x, y)))
    centered_logs = jnp.log(x) - jnp.mean(jnp.log(x))

    assert jnp.allclose(y, centered_logs[:-1])
    assert jnp.allclose(round_trip, x)
    assert jnp.allclose(log_det, simplex_chart_log_det(bijector, x))
    assert jnp.allclose(log_det, -jnp.sum(jnp.log(x)) - jnp.log(3))
