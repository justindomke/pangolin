import math

import jax.numpy as jnp
import pytest

from pangolin import functional, ir
from pangolin.jax_backend import lower, require_array


@pytest.mark.parametrize(
    ("op", "parent_values", "x", "expected_y", "expected_log_det"),
    [
        (ir.Exponential(), (1.0,), 2.0, math.log(2.0), -math.log(2.0)),
        (
            ir.Beta(),
            (2.0, 3.0),
            0.25,
            math.log(0.25 / 0.75),
            -math.log(0.25) - math.log(0.75),
        ),
        (
            ir.Uniform(),
            (1.0, 5.0),
            2.0,
            math.log(0.25 / 0.75),
            math.log(4.0) - math.log(1.0) - math.log(3.0),
        ),
    ],
)
def test_compiled_transforms_round_trip_and_compute_forward_jacobian(
    op, parent_values, x, expected_y, expected_log_det
):
    parents = tuple(ir.RV(ir.Constant(value)) for value in parent_values)
    reconstructed_rv = functional.unconstrain(ir.RV(op, *parents), ())[0]
    free_rv = reconstructed_rv.parents[0]
    assert isinstance(free_rv.op, ir.Bijected)
    bijector = free_rv.op.bijector
    bijector_values = parent_values[: bijector.n_biject_params]

    y = require_array(lower(bijector.forward, (x, *bijector_values)))
    reconstructed = require_array(lower(bijector.inverse, (y, *bijector_values)))
    log_det = require_array(lower(bijector.log_det_jac, (x, y, *bijector_values)))

    assert jnp.allclose(y, expected_y)
    assert jnp.allclose(reconstructed, x)
    assert jnp.allclose(log_det, expected_log_det)
