"""Vectorized loops preserve joint densities when composed with other graph builders."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import distributions as dist

from pangolin import dag, interface as pi, ir, jax_backend
from pangolin.functional import loop


def check_density(outputs, values, reference):
    density = jax_backend.ancestor_log_prob(outputs)
    expected = reference(values)
    np.testing.assert_allclose(density(values), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(jax.jit(density)(values), expected, rtol=1e-6, atol=1e-6)
    actual_grad = jax.jit(jax.grad(density))(values)
    expected_grad = jax.grad(reference)(values)
    for actual, expected in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad)):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("layout", ["vector", "outer", "row", "matrix", "transposed"])
def test_random_location_and_scale_indexing(layout):
    """The five vfor random-parent regressions, including their joint density."""
    a_shape = (2, 3) if layout == "matrix" else (2,)
    b_shape = {
        "vector": (2,),
        "outer": (3,),
        "row": (2, 3),
        "matrix": (2, 3),
        "transposed": (3, 2),
    }[layout]
    a = (
        loop(lambda i, j: pi.normal(0, 10), interface=pi.default, i=2, j=3)
        if len(a_shape) == 2
        else loop(lambda i: pi.normal(0, 10), interface=pi.default, i=2)
    )
    b = (
        loop(lambda i, j: pi.lognormal(0, 1), interface=pi.default, i=b_shape[0], j=b_shape[1])
        if len(b_shape) == 2
        else loop(lambda i: pi.lognormal(0, 1), interface=pi.default, i=b_shape[0])
    )

    if layout == "vector":
        c = loop(lambda i: pi.normal(a[i], b[i]), interface=pi.default, i=2)
        expected_op = ir.VMap(ir.Normal(), (0, 0), 2)
    else:

        def model(i, j):
            location = a[i, j] if layout == "matrix" else a[i]
            scale = b[j] if layout == "outer" else b[j, i] if layout == "transposed" else b[i, j]
            return pi.normal(location, scale)

        c = loop(model, interface=pi.default, i=2, j=3)
        inner_axes = (0 if layout == "matrix" else None, 0)
        outer_axes = (0, None if layout == "outer" else 1 if layout == "transposed" else 0)
        expected_op = ir.VMap(ir.VMap(ir.Normal(), inner_axes, 3), outer_axes, 2)
    assert c.op == expected_op
    assert c.parents == (a, b)
    assert c.shape == ((2,) if layout == "vector" else (2, 3))
    assert len([rv for rv in dag.upstream_nodes(c) if rv.op.random]) == 3

    a_value = jnp.arange(np.prod(a_shape), dtype=float).reshape(a_shape) / 5
    b_value = 1 + jnp.arange(np.prod(b_shape), dtype=float).reshape(b_shape) / 4
    c_value = jnp.arange(np.prod(c.shape), dtype=float).reshape(c.shape) / 3

    def reference(values):
        av, bv, cv = values
        location = av if layout in ("vector", "matrix") else av[:, None]
        scale = bv.T if layout == "transposed" else bv
        return (
            dist.Normal(0, 10).log_prob(av).sum()
            + dist.LogNormal(0, 1).log_prob(bv).sum()
            + dist.Normal(location, scale).log_prob(cv).sum()
        )

    check_density((a, b, c), (a_value, b_value, c_value), reference)


def test_logistic_regression_shares_coefficients():
    X_values = jnp.array([[10.0, 0.1], [-5.0, 0.02], [5.0, 0.19]])
    X = pi.constant(X_values)
    z = loop(lambda i: pi.normal(0, 1), interface=pi.default, i=2)
    y = loop(lambda n: pi.bernoulli_logit(X[n, :] @ z), interface=pi.default, n=3)
    assert y.op == ir.VMap(ir.BernoulliLogit(), (0,), 3)
    (logits,) = y.parents
    assert logits.op == ir.VMap(ir.Matmul(), (0, None), 3)
    assert logits.parents == (X, z)
    density = jax.jit(jax_backend.ancestor_log_prob((z, y)))
    z_value, y_value = jnp.array([0.2, -0.3]), jnp.array([1, 0, 1])
    expected = dist.Normal(0, 1).log_prob(z_value).sum()
    expected += dist.Bernoulli(logits=X_values @ z_value).log_prob(y_value).sum()
    np.testing.assert_allclose(density((z_value, y_value)), expected, rtol=1e-6)


def test_iid_bernoulli_density():
    y = loop(lambda i: pi.bernoulli(0.3), interface=pi.default, i=5)
    assert y.op == ir.VMap(ir.Bernoulli(), (None,), 5)
    assert y.shape == (5,)
    values = jnp.array([0, 1, 0, 0, 1])
    density = jax.jit(jax_backend.ancestor_log_prob(y))
    np.testing.assert_allclose(density(values), dist.Bernoulli(0.3).log_prob(values).sum())


def test_two_independent_random_nodes_in_single_output():
    def model(i):
        a = pi.normal(0, 1)
        b = pi.normal(1, 2)
        return a + b

    output = loop(model, interface=pi.default, i=3)
    a, b = output.parents
    assert a is not b
    assert a.op == b.op == ir.VMap(ir.Normal(), (None, None), 3)
    assert output.op == ir.VMap(ir.Add(), (0, 0), 3)
    check_density(
        (a, b),
        (jnp.array([0.1, 0.2, 0.3]), jnp.array([1.0, 2.0, 3.0])),
        lambda v: dist.Normal(0, 1).log_prob(v[0]).sum() + dist.Normal(1, 2).log_prob(v[1]).sum(),
    )


def test_loop_inside_sfor_matches_parallel_sfor_axis():
    compact = pi.sfor(lambda previous, t, i: pi.normal(previous[i], 1), init=np.zeros(2), t=3, i=2)
    explicit = pi.sfor(
        lambda previous, t: loop(lambda i: pi.normal(previous[i], 1), interface=pi.default, i=2),
        init=np.zeros(2),
        t=3,
    )
    expected = ir.Scan(
        ir.Composite(1, (ir.Constant(1), ir.VMap(ir.Normal(), (0, None), 2)), ((), (0, 1))), 3, ()
    )
    assert compact.op == explicit.op == expected
    values = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    reference = (
        lambda v: dist.Normal(jnp.concatenate((jnp.zeros((1, 2)), v[:-1])), 1).log_prob(v).sum()
    )
    check_density(explicit, values, reference)


def test_sfor_inside_loop_builds_vmap_of_scan():
    x = loop(
        lambda i: pi.sfor(lambda previous, t: pi.normal(previous + i, 1), init=0.0, t=3),
        interface=pi.default,
        i=2,
    )
    expected_scan = ir.Scan(
        ir.Composite(2, (ir.Add(), ir.Constant(1), ir.Normal()), ((0, 1), (), (2, 3))), 3, (None,)
    )
    assert x.shape == (2, 3)
    assert x.op == ir.VMap(expected_scan, (None, 0), 2)
    assert x.parents[0].op == ir.Constant(0.0)
    assert x.parents[1].op == ir.Constant([0, 1])
    values = jnp.array([[0.1, 0.3, 0.4], [1.0, 2.1, 2.9]])
    reference = (
        lambda v: dist.Normal(
            jnp.concatenate((jnp.zeros((2, 1)), v[:, :-1]), axis=1) + jnp.arange(2)[:, None], 1
        )
        .log_prob(v)
        .sum()
    )
    check_density(x, values, reference)


def test_loop_sfor_loop_with_shared_carry():
    def individual(i):
        def step(previous, t):
            return loop(lambda j: pi.normal(previous[j] + i + t, 1), interface=pi.default, j=2)

        return pi.sfor(step, init=np.zeros(2), t=3)

    x = loop(individual, interface=pi.default, i=2)
    assert x.shape == (2, 3, 2)
    values = jnp.arange(12, dtype=float).reshape(2, 3, 2) / 10

    def reference(v):
        previous = jnp.concatenate((jnp.zeros((2, 1, 2)), v[:, :-1, :]), axis=1)
        means = previous + jnp.arange(2)[:, None, None] + jnp.arange(3)[None, :, None]
        return dist.Normal(means, 1).log_prob(v).sum()

    check_density(x, values, reference)


def test_multivariate_random_output_retains_event_axes():
    covariance = jnp.array([[1.0, 0.3], [0.3, 2.0]])
    y = loop(
        lambda i: pi.multi_normal(pi.constant([0.0, 1.0]) + i, covariance),
        interface=pi.default,
        i=3,
    )
    assert y.shape == (3, 2)
    assert y.op == ir.VMap(ir.MultiNormal(), (0, None), 3)
    values = jnp.array([[0.1, 0.2], [1.0, 2.0], [2.0, 3.0]])
    reference = (
        lambda v: dist.MultivariateNormal(
            jnp.array([0.0, 1.0]) + jnp.arange(3)[:, None], covariance_matrix=covariance
        )
        .log_prob(v)
        .sum()
    )
    check_density(y, values, reference)
