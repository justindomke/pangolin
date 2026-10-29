"""Recursive truncation transforms, Jacobians, and densities compiled to JAX."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pangolin import functional, interface as pi, ir
from pangolin.jax_backend import (
    ancestor_sample_flat,
    ancestor_sample_graph,
    lower_deterministic,
    lower_random,
)


def check_transform(rv, value):
    """Compare the compiled change of variables with autodiff and the original density."""
    reconstructed, _ = functional.unconstrain(rv, ())
    free = reconstructed.parents[0]
    assert isinstance(free.op, ir.Bijected)
    bijector = free.op.bijector
    parents = ancestor_sample_flat(free.parents, None)
    params = parents[: bijector.n_biject_params]

    def forward(x):
        return lower_deterministic(bijector.forward, (x, *params))

    def inverse(y):
        return lower_deterministic(bijector.inverse, (y, *params))

    x = jnp.asarray(value)
    y = jax.jit(forward)(x)
    np.testing.assert_allclose(jax.jit(inverse)(y), x, rtol=2e-5, atol=2e-5)
    jacobian = jax.jacfwd(lambda z: forward(z.reshape(x.shape)).reshape(-1))(x.reshape(-1))
    expected_log_det = jnp.linalg.slogdet(jacobian).logabsdet
    log_det = lower_deterministic(bijector.log_det_jac, (x, y, *params))
    np.testing.assert_allclose(log_det, expected_log_det, rtol=2e-5, atol=2e-5)
    base = lower_random(rv.op, ancestor_sample_flat(rv.parents, None))
    transformed = lower_random(free.op, parents)
    np.testing.assert_allclose(
        transformed.log_prob(y), base.log_prob(x) - log_det, rtol=3e-5, atol=3e-5
    )
    assert jnp.all(jnp.isfinite(jax.grad(lambda z: transformed.log_prob(z))(y)))
    if bijector.sign is not None:
        sign = lower_deterministic(bijector.sign, params)
        np.testing.assert_allclose(sign, jnp.sign(jacobian[0, 0]))
    return forward, inverse


@pytest.mark.parametrize("lo, hi, value", [(-0.5, 1.5, 0.3), (0.2, None, 0.8), (None, 1.2, -0.1)])
def test_normal(lo, hi, value):
    forward, inverse = check_transform(pi.truncated(pi.normal, lo, hi)(0.1, 1.1), value)
    expected = (
        np.log((value - lo) / (hi - value))
        if lo is not None and hi is not None
        else np.log(value - lo) if lo is not None else np.log(hi - value)
    )
    np.testing.assert_allclose(forward(jnp.array(value)), expected, rtol=2e-5, atol=2e-5)
    draws = jax.vmap(inverse)(jnp.array([-8.0, 0.0, 8.0]))
    assert jnp.all(jnp.isfinite(draws))
    if lo is not None:
        assert jnp.all(draws > lo)
    if hi is not None:
        assert jnp.all(draws < hi)


@pytest.mark.parametrize("lo, hi, value", [(0.3, None, 0.7), (None, 1.8, 0.7)])
def test_positive_base_preserves_the_unspecified_endpoint(lo, hi, value):
    _, inverse = check_transform(pi.truncated(pi.exponential, lo, hi)(1.2), value)
    values = jax.vmap(inverse)(jnp.array([-5.0, -1.0, 1.0]))
    assert jnp.all(values > (0.0 if lo is None else lo))
    if hi is not None:
        assert jnp.all(values < hi)


@pytest.mark.parametrize("lo, hi", [(0.2, None), (None, 0.8)])
def test_bounded_base_preserves_the_unspecified_endpoint(lo, hi):
    _, inverse = check_transform(pi.truncated(pi.uniform, lo, hi)(0.0, 1.0), 0.5)
    values = jax.vmap(inverse)(jnp.array([-4.0, 0.0, 1.0]))
    assert jnp.all(values > (0.0 if lo is None else lo))
    assert jnp.all(values < (1.0 if hi is None else hi))


@pytest.mark.parametrize("factor", [-2.0, 2.0])
@pytest.mark.parametrize("lower", [False, True])
def test_parameter_dependent_bijection_sign(factor, lower):
    model = pi.tforms.scale_by(pi.constant(factor))(pi.tforms.exp(pi.normal))
    lo, hi = (factor * 0.8, None) if lower else (None, factor * 0.8)
    # Choose a point on the appropriate side, independent of the sign of factor.
    value = factor * 0.8 + (0.2 if lower else -0.2)
    check_transform(pi.truncated(model, lo, hi)(0.0, 0.8), value)


def test_same_jitted_graph_handles_both_signs_and_differentiates_parent_values():
    factor, lo, value = [pi.constant(x) for x in (2.0, 0.2, 0.7)]
    rv = pi.truncated(pi.tforms.scale_by(factor)(pi.normal), lo)(0.0, 1.0)
    transform = functional.transform_for(rv.op, rv.parents)
    assert transform is not None
    forward = transform.forward(value)
    restored = transform.inverse(forward)
    log_det = transform.log_det_jac(value, forward)

    @jax.jit
    def evaluate(a, b, x):
        return ancestor_sample_graph(
            (forward, restored, log_det), None, {factor: a, lo: b, value: x}
        )

    factors = jnp.array([-3.0, -2.0, 2.0, 3.0])
    outputs, restored_values, log_dets = jax.vmap(evaluate, (0, None, None))(factors, 0.1, 0.9)
    np.testing.assert_allclose(outputs, jnp.log(0.8 / jnp.abs(factors)), rtol=2e-5)
    np.testing.assert_allclose(restored_values, 0.9, rtol=2e-5)
    np.testing.assert_allclose(log_dets, -jnp.log(0.8), rtol=2e-5)
    for a in (-2.0, 2.0):
        derivatives = jax.grad(lambda a, b, x: evaluate(a, b, x)[0], argnums=(0, 1, 2))(a, 0.1, 0.9)
        np.testing.assert_allclose(derivatives, (-1 / a, -1 / 0.8, 1 / 0.8), rtol=2e-5)


@pytest.mark.parametrize("model", [pi.gamma, pi.beta])
def test_unconstraining_does_not_require_quantile_evaluation(model):
    check_transform(pi.truncated(model, 0.2)(2.0, 3.0), 0.4)


@pytest.mark.parametrize("outer_lower", [False, True])
def test_nested_truncation(outer_lower):
    inner = pi.truncated(pi.normal, -1.0, 2.0)
    outer = pi.truncated(inner, 0.0, None) if outer_lower else pi.truncated(inner, None, 1.0)
    check_transform(outer(0.1, 0.9), 0.4)


def test_one_sided_truncation_of_one_sided_truncation():
    model = pi.truncated(pi.truncated(pi.exponential, None, 2.0), 0.3)
    check_transform(model(1.0), 0.8)


def test_unbounded_truncation_delegates_to_base():
    value = ir.RV(ir.Truncated(ir.Exponential(), lower=False, upper=False), pi.constant(1.3))
    check_transform(value, 0.8)


def test_composite_with_computed_bounds():
    @pi.composite
    def model(location):
        return pi.truncated(pi.normal, location - 0.5)(location, 0.7)

    check_transform(model(pi.constant(0.3)), 0.5)


def test_nested_vmap_with_varying_bounds():
    model = pi.vmap(pi.vmap(lambda lo: pi.truncated(pi.normal, lo)(0.0, 1.0)))
    check_transform(model(pi.constant([[-0.4, -0.2], [0.0, 0.2]])), [[0.1, 0.3], [0.5, 0.7]])


def test_scan_with_carry_dependent_truncation():
    def step(previous):
        return pi.truncated(pi.normal, previous - 0.5)(previous, 0.6)

    check_transform(pi.scan(step, 3)(pi.constant(0.2)), [0.4, 0.1, 0.5])


@pytest.mark.parametrize("lo, hi", [(0.0, 1.0), (-0.5, None), (0.8, 0.3), (0.3, np.inf)])
def test_invalid_bounds_are_still_rejected_by_the_backend(lo, hi):
    rv = pi.truncated(pi.exponential, lo, hi)(1.0)
    reconstructed, _ = functional.unconstrain(rv, ())
    free = reconstructed.parents[0]
    lowered = lower_random(free.op, ancestor_sample_flat(free.parents, None))
    assert jnp.isnan(lowered.log_prob(jnp.array(0.0)))
