from pangolin.jax_backend import (
    ancestor_log_prob_flat,
    ancestor_sample_flat,
    lower,
    lower_deterministic,
    require_array,
    require_distribution,
)
import numpy as np
from pangolin import ir

import jax
import jax.numpy as jnp
import numpyro.distributions


def test_nested_composite_quantile_endpoints():
    # Inner distribution: Uniform(lower, lower + exp(width)). Outer reverses its inputs.
    inner = ir.Composite(2, (ir.Exp(), ir.Add(), ir.Uniform()), ((0,), (1, 2), (1, 3)))
    outer = ir.Composite(2, (inner,), ((1, 0),))

    def bounds(lower, width):
        parents = (lower, width)
        return jnp.stack(
            (
                lower_deterministic(ir.Quantile(outer), (0.0, *parents)),
                lower_deterministic(ir.Quantile(outer), (1.0, *parents)),
            )
        )

    lower = jnp.array([-3.0, 0.0, 2.0])
    width = jnp.array([0.0, 1.0, -1.0])
    actual = jax.jit(jax.vmap(bounds))(lower, width)
    expected = jnp.stack((lower, lower + jnp.exp(width)), axis=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    derivative = jax.jit(jax.jacfwd(bounds, (0, 1)))(lower[0], width[0])
    np.testing.assert_allclose(derivative, ([1.0, 1.0], [0.0, 1.0]))


def test_add():
    # x -> x+x

    op = ir.Composite(1, (ir.Add(),), [[0, 0]])

    out = require_array(lower(op, [1.5]))
    expected = 3.0
    assert np.allclose(out, expected)

    x = ir.RV(ir.Constant(1.5))
    y = ir.RV(op, x)
    assert isinstance(y.op, ir.Composite)

    [out] = ancestor_sample_flat([y], None)
    assert np.allclose(expected, out)


def test_add_mul():
    # x,y -> (x+x)*y
    op = ir.Composite(
        2,
        (
            ir.Add(),
            ir.Mul(),
        ),
        [[0, 0], [2, 1]],
    )

    x = ir.RV(ir.Constant(3.3))
    y = ir.RV(ir.Constant(4.4))
    z = ir.RV(op, x, y)
    assert isinstance(z.op, ir.Composite)

    expected = (3.3 + 3.3) * 4.4
    out = require_array(lower(z.op, [3.3, 4.4]))
    assert np.allclose(expected, out)

    [out] = ancestor_sample_flat([z], None)
    assert np.allclose(expected, out)


def test_bernoulli():
    op = ir.Composite(1, (ir.Bernoulli(),), [[0]])

    x = ir.RV(ir.Constant(0.7))
    y = ir.RV(op, x)
    assert isinstance(y.op, ir.Composite)

    key = jax.random.PRNGKey(0)
    value = require_distribution(lower(y.op, [0.7])).sample(key)
    l = require_distribution(lower(y.op, [0.7])).log_prob(value)

    numpyro_dist = numpyro.distributions.Bernoulli(0.7)
    assert isinstance(numpyro_dist, numpyro.distributions.Distribution)
    expected = numpyro_dist.log_prob(value)

    assert np.allclose(l, expected)


def test_exponential():
    op = ir.Composite(1, (ir.Exponential(),), [[0]])

    x = ir.RV(ir.Constant(0.7))
    y = ir.RV(op, x)
    assert isinstance(y.op, ir.Composite)

    key = jax.random.PRNGKey(0)
    value = require_distribution(lower(y.op, [0.7])).sample(key)
    l = require_distribution(lower(y.op, [0.7])).log_prob(value)

    numpyro_dist = numpyro.distributions.Exponential(0.7)
    expected = numpyro_dist.log_prob(value)

    assert np.allclose(l, expected)


def test_add_normal():
    # z ~ Normal(x+y, y)
    op = ir.Composite(
        2,
        (
            ir.Add(),
            ir.Normal(),
        ),
        [[0, 1], [2, 1]],
    )

    x = ir.RV(ir.Constant(0.3))
    y = ir.RV(ir.Constant(0.1))
    z = ir.RV(op, x, y)

    key = jax.random.PRNGKey(0)

    value = require_distribution(lower(z.op, [0.3, 0.1])).sample(jax.random.split(key)[1])
    [value2] = ancestor_sample_flat([z], key)

    assert value == value2

    l = require_distribution(lower(z.op, [0.3, 0.1])).log_prob(value)
    l2 = ancestor_log_prob_flat([z], [value])

    tmp = 0.3 + 0.1
    numpyro_dist = numpyro.distributions.Normal(tmp, 0.1)
    expected = numpyro_dist.log_prob(value)
    assert np.allclose(l, expected)
    assert np.allclose(l2, expected)
