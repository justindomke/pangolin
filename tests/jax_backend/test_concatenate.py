import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pangolin import functional, ir
from pangolin.jax_backend import (
    ancestor_log_prob,
    ancestor_sample_graph,
    lower,
    require_array,
)


@pytest.mark.parametrize(
    ("axis", "shapes"),
    [
        (0, [(2,), (3,), (1,)]),
        (0, [(1, 3), (2, 3)]),
        (1, [(2, 1), (2, 3)]),
        (-1, [(2, 1), (2, 3)]),
        (-2, [(1, 3), (2, 3)]),
        (0, [(2, 3)]),
    ],
)
def test_concatenate_matches_numpy(axis, shapes):
    parents = [np.arange(np.prod(shape)).reshape(shape) for shape in shapes]
    actual = lower(ir.Concatenate(axis), parents)
    assert isinstance(actual, jax.Array)
    assert np.array_equal(actual, np.concatenate(parents, axis=axis))


def test_concatenate_is_jittable_and_differentiable():
    def concatenate(a, b):
        return require_array(lower(ir.Concatenate(), (a, b)))

    a, b = jnp.array([1.0, 2.0]), jnp.array([3.0])
    assert jnp.array_equal(jax.jit(concatenate)(a, b), jnp.array([1.0, 2.0, 3.0]))
    gradients = jax.grad(lambda a, b: jnp.sum(concatenate(a, b) ** 2), argnums=(0, 1))(a, b)
    assert jnp.array_equal(gradients[0], 2 * a)
    assert jnp.array_equal(gradients[1], 2 * b)


@pytest.mark.parametrize(
    ("parents", "message"),
    [
        ([], "at least one parent"),
        ([1.0, 2.0], "out of bounds"),
        ([np.ones((2, 3)), np.ones((2,))], "same number of dimensions"),
        ([np.ones((2, 3)), np.ones((2, 4))], "must match except along axis"),
    ],
)
def test_concatenate_uses_ir_shape_validation(parents, message):
    with pytest.raises(ValueError, match=message):
        lower(ir.Concatenate(), parents)


def test_unconstrained_scan_with_carry_dependent_bounds():
    # Each step is Uniform(previous, previous + 1).
    base = ir.Composite(1, (ir.Constant(1.0), ir.Add(), ir.Uniform()), ((), (0, 1), (0, 2)))
    original = ir.RV(ir.Scan(base, 3, ()), ir.RV(ir.Constant(0.0)))
    reconstructed = functional.unconstrain(original, ())[0]
    free = reconstructed.parents[0]
    assert isinstance(free.op, ir.Bijected)

    values = jnp.array([-0.4, 0.2, 0.8])
    actual = jax.jit(lambda u: ancestor_sample_graph(reconstructed, None, boundary={free: u}))(
        values
    )
    assert jnp.allclose(actual, jnp.cumsum(jax.nn.sigmoid(values)))

    log_prob = ancestor_log_prob(free)
    expected = jnp.sum(jax.nn.log_sigmoid(values) + jax.nn.log_sigmoid(-values))
    assert jnp.allclose(jax.jit(log_prob)(values), expected)
    assert jnp.allclose(jax.grad(log_prob)(values), 1 - 2 * jax.nn.sigmoid(values))
    assert jnp.all(jnp.isfinite(ancestor_sample_graph(free, jax.random.PRNGKey(0))))
