"""Scalar quantiles, recursive distribution wrappers, and deterministic compositions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike
from scipy import stats

from pangolin import ir
from pangolin.jax_backend import ancestor_sample_graph, lower_deterministic


@pytest.mark.parametrize(
    "base,parents,reference",
    [
        (ir.Normal(), (0.3, 0.8), stats.norm(0.3, 0.8)),
        (ir.NormalPrec(), (0.3, 4.0), stats.norm(0.3, 0.5)),
        (ir.Cauchy(), (0.3, 0.8), stats.cauchy(0.3, 0.8)),
        (ir.Exponential(), (2.0,), stats.expon(scale=0.5)),
        (ir.Lognormal(), (0.3, 0.8), stats.lognorm(0.8, scale=np.exp(0.3))),
        (ir.Uniform(), (-1.0, 2.0), stats.uniform(-1.0, 3.0)),
    ],
)
def test_primitives_against_scipy_including_endpoints_and_invalid_probabilities(
    base: ir.Op, parents: tuple[ArrayLike, ...], reference
):
    probabilities = jnp.array(
        [-np.inf, -0.1, 0, 0.001, 0.2, 0.5, 0.9, 0.999, 1, 1.1, np.inf, np.nan]
    )
    evaluate = lambda p: lower_deterministic(ir.Quantile(base), (p, *parents))
    expected = reference.ppf(np.asarray(probabilities))
    actual = jax.jit(jax.vmap(evaluate))(probabilities)
    # NumPyro's tangent-based Cauchy quantile loses precision near the float32 tails.
    rtol = 5e-5 if isinstance(base, ir.Cauchy) else 3e-5
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=1e-6, equal_nan=True)
    for p, result in zip(probabilities, expected):
        np.testing.assert_allclose(evaluate(p), result, rtol=rtol, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize(
    "base,parents",
    [
        (ir.Beta(), (2.0, 3.0)),
        (ir.Gamma(), (2.0, 3.0)),
        (ir.StudentT(), (5.0, 0.3, 0.8)),
        (ir.Bernoulli(), (0.3,)),
        (ir.BernoulliLogit(), (0.3,)),
        (ir.Binomial(), (5, 0.3)),
        (ir.BetaBinomial(), (5, 2.0, 3.0)),
        (ir.Categorical(), (jnp.array([0.2, 0.8]),)),
        (ir.Poisson(), (2.0,)),
    ],
)
def test_unsupported_bases_raise_explicitly(base: ir.Op, parents: tuple[ArrayLike, ...]):
    with pytest.raises(
        NotImplementedError, match=f"Quantile is not implemented for {type(base).__name__}"
    ):
        lower_deterministic(ir.Quantile(base), (0.3, *parents))


def test_unknown_distribution_is_not_treated_as_normal():
    class Unknown(ir.Op):
        random: bool = True
        """True"""
        discrete: bool = False
        """False"""

        def get_shape(self) -> ir.Shape:
            return ()

    with pytest.raises(NotImplementedError, match="Quantile is not implemented for Unknown"):
        lower_deterministic(ir.Quantile(Unknown()), (0.3,))


def test_composite_with_reordered_repeated_and_unused_inputs():
    base = ir.Composite(3, (ir.Mul(), ir.Exp(), ir.Normal()), ((1, 1), (3,), (0, 4)))
    actual = lower_deterministic(ir.Quantile(base), (0.7, 0.3, 0.8, jnp.ones(4)))
    np.testing.assert_allclose(actual, stats.norm.ppf(0.7, 0.3, np.exp(0.8**2)), rtol=1e-6)


def test_zero_input_and_single_op_composites():
    base = ir.Composite(0, (ir.Constant(0.0), ir.Constant(1.0), ir.Normal()), ((), (), (0, 1)))
    assert lower_deterministic(ir.Quantile(base), (0.5,)) == 0.0
    reordered = ir.Composite(2, (ir.Normal(),), ((1, 0),))
    np.testing.assert_allclose(
        lower_deterministic(ir.Quantile(reordered), (0.7, 0.8, 0.3)),
        stats.norm.ppf(0.7, 0.3, 0.8),
        rtol=1e-6,
    )


def exp_bijector(decreasing: bool = False) -> ir.Bijector:
    if decreasing:
        forward = ir.Composite(1, (ir.Exp(), ir.Constant(-1.0), ir.Mul()), ((0,), (), (1, 2)))
        inverse = ir.Composite(1, (ir.Constant(-1.0), ir.Mul(), ir.Log()), ((), (0, 1), (2,)))
    else:
        forward, inverse = ir.Exp(), ir.Log()
    return ir.Bijector(
        forward,
        inverse,
        ir.Composite(2, (ir.Identity(),), ((0,),)),
        sign=ir.Constant(-1 if decreasing else 1),
    )


@pytest.mark.parametrize("decreasing", [False, True])
@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.8, 1.0, -0.1, 1.1, np.nan])
def test_bijected_orientation_endpoints_and_invalid_probabilities(decreasing: bool, p: float):
    base = ir.Bijected(ir.Normal(), exp_bijector(decreasing))
    actual = jax.jit(lambda q: lower_deterministic(ir.Quantile(base), (q, 0.3, 0.8)))(p)
    expected = (
        -stats.lognorm.isf(p, 0.8, scale=np.exp(0.3))
        if decreasing
        else stats.lognorm.ppf(p, 0.8, scale=np.exp(0.3))
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, equal_nan=True)


@pytest.mark.parametrize("decreasing", [False, True])
def test_base_support_smaller_than_bijector_domain(decreasing: bool):
    base = ir.Bijected(ir.Uniform(), exp_bijector(decreasing))
    for p in [0.0, 0.3, 1.0]:
        actual = lower_deterministic(ir.Quantile(base), (p, 1.0, 2.0))
        expected = -np.exp(2 - p) if decreasing else np.exp(1 + p)
        np.testing.assert_allclose(actual, expected, rtol=2e-6)


def test_bijection_is_evaluated_directly_without_endpoint_limits():
    # x / (1 + x) tends to one at infinity; direct evaluation would be inf / inf.
    forward = ir.Composite(1, (ir.Constant(1.0), ir.Add(), ir.Div()), ((), (0, 1), (0, 2)))
    inverse = ir.Composite(1, (ir.Constant(1.0), ir.Sub(), ir.Div()), ((), (1, 0), (0, 2)))
    ldj = ir.Composite(
        2,
        (ir.Constant(1.0), ir.Add(), ir.Log(), ir.Constant(-2.0), ir.Mul()),
        ((), (0, 2), (3,), (), (4, 5)),
    )
    bijector = ir.Bijector(
        forward,
        inverse,
        ldj,
        sign=ir.Constant(1),
    )
    base = ir.Bijected(ir.Exponential(), bijector)
    evaluate = lambda p, rate: lower_deterministic(ir.Quantile(base), (p, rate))
    probabilities = jnp.array([0.1, 0.4, 0.8])
    expected_x = stats.expon.ppf(probabilities, scale=0.5)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(evaluate, (0, None)))(probabilities, 2.0),
        expected_x / (1 + expected_x),
        rtol=2e-6,
    )
    # No endpoint-limit machinery replaces the forward map's inf / inf.
    assert jnp.isnan(jax.jit(evaluate)(1.0, 2.0))


def test_parameter_dependent_sign_under_vmap_and_grad():
    sign = ir.Composite(1, (ir.Abs(), ir.Div()), ((0,), (0, 1)))
    ldj = ir.Composite(3, (ir.Abs(), ir.Log()), ((2,), (3,)))
    bijector = ir.Bijector(ir.Mul(), ir.Div(), ldj, 1, sign=sign)
    base = ir.Bijected(ir.Normal(), bijector)
    evaluate = lambda scale: lower_deterministic(ir.Quantile(base), (0.3, scale, 0.2, 0.8))
    scales = jnp.array([-2.0, 3.0])
    expected = np.asarray(scales) * 0.2 + np.abs(scales) * 0.8 * stats.norm.ppf(0.3)
    np.testing.assert_allclose(jax.jit(jax.vmap(evaluate))(scales), expected, rtol=2e-6)
    expected_grad = 0.2 + np.sign(scales) * 0.8 * stats.norm.ppf(0.3)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(jax.grad(evaluate)))(scales), expected_grad, rtol=2e-6
    )


def test_two_decreasing_bijections_restore_original_quantile():
    negative = ir.Composite(1, (ir.Constant(-1.0), ir.Mul()), ((), (0, 1)))
    bijector = ir.Bijector(
        negative,
        negative,
        ir.Composite(2, (ir.Constant(0.0),), ((),)),
        sign=ir.Constant(-1),
    )
    base = ir.Bijected(ir.Bijected(ir.Exponential(), bijector), bijector)
    probabilities = jnp.array([0.0, 0.2, 0.8, 1.0])
    actual = jax.jit(jax.vmap(lambda p: lower_deterministic(ir.Quantile(base), (p, 2.0))))(
        probabilities
    )
    np.testing.assert_allclose(actual, stats.expon.ppf(probabilities, scale=0.5), rtol=1e-6)


def test_transformed_finite_support_endpoint_gradients():
    base = ir.Bijected(ir.Uniform(), exp_bijector())
    evaluate = lambda p, lo, hi: lower_deterministic(ir.Quantile(base), (p, lo, hi))
    derivative = jax.jit(jax.vmap(jax.grad(evaluate, (1, 2)), (0, None, None)))
    lo_derivative, hi_derivative = derivative(jnp.array([0.0, 1.0]), 1.0, 2.0)
    np.testing.assert_allclose(lo_derivative, [np.e, 0.0], rtol=1e-6)
    np.testing.assert_allclose(hi_derivative, [0.0, np.e**2], rtol=1e-6)


def test_nested_bijected_composite_and_nonleading_axis_vmap():
    normal = ir.Composite(2, (ir.Exp(), ir.Normal()), ((1,), (0, 2)))
    lognormal = ir.Bijected(normal, exp_bijector())
    base = ir.Composite(2, (ir.Bijected(lognormal, exp_bijector()),), ((0, 1),))
    op = ir.VMap(ir.VMap(ir.Quantile(base), (0, 0, None)), (1, 1, None))
    probabilities = jnp.array([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])
    locations = jnp.array([[-0.3, -0.2, -0.1], [0.0, 0.1, 0.2]])
    actual = jax.jit(lambda p, loc: lower_deterministic(op, (p, loc, -0.5)))(
        probabilities, locations
    )
    expected = np.exp(np.exp(stats.norm.ppf(probabilities.T, locations.T, np.exp(-0.5))))
    np.testing.assert_allclose(actual, expected, rtol=3e-6)


def test_quantile_inside_scan():
    step = ir.Composite(2, (ir.Quantile(ir.Normal()),), ((1, 0, 1),))
    op = ir.Scan(step, 3, (None,))
    actual = lower_deterministic(op, (0.0, 0.8))
    expected = np.arange(1, 4) * 0.8 * stats.norm.ppf(0.8)
    np.testing.assert_allclose(actual, expected, rtol=2e-6)


def test_vjp_derivatives_and_cdf_inverse_identity():
    quantile = ir.Quantile(ir.Normal())
    p, loc, scale = 0.3, 0.2, 0.8
    x = lower_deterministic(quantile, (p, loc, scale))
    np.testing.assert_allclose(
        lower_deterministic(ir.CDF(ir.Normal()), (x, loc, scale)), p, rtol=1e-6
    )
    for argnum, expected in enumerate([1 / stats.norm.pdf(x, loc, scale), 1.0, stats.norm.ppf(p)]):
        evaluate = lambda q: lower_deterministic(ir.VJP(quantile, argnum), (1.0, q, loc, scale))
        np.testing.assert_allclose(jax.jit(evaluate)(p), expected, rtol=2e-6)


def test_missing_bijector_sign():
    bijector = ir.Bijector(ir.Exp(), ir.Log(), ir.Composite(2, (ir.Identity(),), ((0,),)))
    with pytest.raises(NotImplementedError, match="without sign"):
        lower_deterministic(ir.Quantile(ir.Bijected(ir.Normal(), bijector)), (0.3, 0.0, 1.0))


def test_invalid_shapes_and_complex_probability():
    with pytest.raises(ValueError, match="probability must be scalar"):
        lower_deterministic(ir.Quantile(ir.Normal()), (jnp.array([0.3]), 0.0, 1.0))
    with pytest.raises(ValueError, match="base_op must have scalar output"):
        lower_deterministic(ir.Quantile(ir.MultiNormal()), (0.3, jnp.zeros(2), jnp.eye(2)))
    with pytest.raises(TypeError, match="real probability"):
        lower_deterministic(ir.Quantile(ir.Normal()), (0.3 + 0.1j, 0.0, 1.0))


def test_graph_evaluation_needs_no_sampling_key_when_random_parents_are_supplied():
    zero, one = ir.RV(ir.Constant(0.0)), ir.RV(ir.Constant(1.0))
    probability = ir.RV(ir.Uniform(), zero, one)
    location = ir.RV(ir.Normal(), zero, one)
    result = ir.RV(ir.Quantile(ir.Normal()), probability, location, one)
    actual = ancestor_sample_graph(
        result, None, {probability: jnp.array(0.3), location: jnp.array(0.2)}
    )
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, stats.norm.ppf(0.3, 0.2), rtol=2e-6)
