"""Truncation's shared contract across sampling, density, CDF, and Quantile."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from pangolin import ir
from pangolin.jax_backend import lower_deterministic, lower_random


@pytest.mark.parametrize(
    "lower,upper", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("transformed", [False, True])
def test_all_operations_and_bound_configurations(lower, upper, transformed):
    base = ir.Normal()
    reference = stats.norm(0.3, 0.8)
    lo, hi = -0.5, 1.5
    if transformed:
        bijector = ir.Bijector(
            ir.Exp(), ir.Log(), ir.Composite(2, [ir.Identity()], [[0]]), sign=ir.Constant(1)
        )
        base = ir.Bijected(base, bijector)
        reference = stats.lognorm(0.8, scale=np.exp(0.3))
        lo, hi = 0.4, 3.0
    op = ir.Truncated(base, lower=lower, upper=upper)
    parents = ((lo,) if lower else ()) + ((hi,) if upper else ()) + (0.3, 0.8)
    distribution = lower_random(op, parents)
    a, b = reference.cdf(lo) if lower else 0, reference.cdf(hi) if upper else 1
    probabilities = jnp.array([0.1, 0.3, 0.6, 0.9])
    quantile = lambda p: lower_deterministic(ir.Quantile(op), (p, *parents))
    cdf = lambda x: lower_deterministic(ir.CDF(op), (x, *parents))
    values = jax.jit(jax.vmap(quantile))(probabilities)
    np.testing.assert_allclose(values, reference.ppf(a + probabilities * (b - a)), rtol=3e-6)
    np.testing.assert_allclose(jax.jit(jax.vmap(cdf))(values), probabilities, atol=3e-7)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(distribution.log_prob))(values),
        reference.logpdf(values) - np.log(b - a),
        atol=3e-6,
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(jax.grad(cdf)))(values),
        np.exp(reference.logpdf(values)) / (b - a),
        rtol=3e-6,
    )
    samples = jax.jit(jax.vmap(distribution.sample))(jax.random.split(jax.random.key(31), 2048))
    assert stats.kstest((reference.cdf(samples) - a) / (b - a), "uniform").statistic < 0.04


@pytest.mark.parametrize(
    "lower,upper,bounds",
    [
        (True, True, (0.0, 0.8)),
        (True, True, (0.2, 1.0)),
        (True, True, (-0.1, 0.8)),
        (True, True, (0.2, 1.1)),
        (True, True, (0.8, 0.2)),
        (True, True, (0.5, 0.5)),
        (True, True, (-np.inf, 0.8)),
        (True, True, (0.2, np.inf)),
        (True, True, (np.nan, 0.8)),
        (True, True, (0.2, np.nan)),
        (True, False, (0.0,)),
        (True, False, (-0.1,)),
        (True, False, (np.inf,)),
        (True, False, (-np.inf,)),
        (True, False, (np.nan,)),
        (False, True, (1.0,)),
        (False, True, (1.1,)),
        (False, True, (np.inf,)),
        (False, True, (-np.inf,)),
        (False, True, (np.nan,)),
    ],
)
def test_invalid_bounds_fail_every_interpretation_even_at_outside_queries(lower, upper, bounds):
    op = ir.Truncated(ir.Uniform(), lower=lower, upper=upper)

    def evaluate(*bounds):
        parents = (*bounds, 0.0, 1.0)
        distribution = lower_random(op, parents)
        return jnp.stack(
            (
                distribution.sample(jax.random.key(0)),
                distribution.log_prob(0.5),
                distribution.log_prob(-10.0),
                lower_deterministic(ir.CDF(op), (0.5, *parents)),
                lower_deterministic(ir.CDF(op), (-10.0, *parents)),
                lower_deterministic(ir.CDF(op), (10.0, *parents)),
                lower_deterministic(ir.Quantile(op), (0.5, *parents)),
            )
        )

    assert np.all(np.isnan(evaluate(*bounds)))
    assert np.all(np.isnan(jax.jit(evaluate)(*bounds)))


@pytest.mark.parametrize("lower,upper", [(True, False), (False, True), (True, True)])
def test_outside_truncation_is_masked_before_bijector_inverse(lower, upper):
    bijector = ir.Bijector(
        ir.Exp(), ir.Log(), ir.Composite(2, [ir.Identity()], [[0]]), sign=ir.Constant(1)
    )
    op = ir.Truncated(ir.Bijected(ir.Normal(), bijector), lower=lower, upper=upper)
    bounds = ((0.4,) if lower else ()) + ((3.0,) if upper else ())
    parents = (*bounds, 0.3, 0.8)
    # Only query outside a supplied truncation bound. The other side retains the
    # bijection's interior-only contract.
    value = -1.0 if lower else 10.0
    cdf = lambda x: lower_deterministic(ir.CDF(op), (x, *parents))
    density = lower_random(op, parents).log_prob
    assert jax.jit(cdf)(value) == (0.0 if lower else 1.0)
    assert jax.jit(density)(value) == -jnp.inf
    assert jax.jit(jax.grad(cdf))(value) == 0.0
    assert jax.jit(jax.grad(density))(value) == 0.0
    assert jnp.isnan(cdf(jnp.nan))
    assert jnp.isnan(density(jnp.nan))


@pytest.mark.parametrize(
    "bounds",
    [
        (0.2, 0.8, 0.0, 0.9),  # Invalid inner lower bound cannot be repaired by the outer one.
        (0.2, 0.8, 0.1, 1.0),
        (0.1, 0.8, 0.1, 0.9),  # Outer bound at the inner endpoint is not interior.
        (0.2, 0.9, 0.1, 0.9),
        (0.05, 0.8, 0.1, 0.9),
    ],
)
def test_nested_truncation_cannot_hide_invalid_inner_or_outer_bounds(bounds):
    op = ir.Truncated(ir.Truncated(ir.Uniform()))
    parents = (*bounds, 0.0, 1.0)
    distribution = lower_random(op, parents)
    assert jnp.isnan(jax.jit(distribution.sample)(jax.random.key(0)))
    assert jnp.isnan(jax.jit(distribution.log_prob)(0.5))
    for wrapper in [ir.CDF, ir.Quantile]:
        assert jnp.isnan(jax.jit(lambda p: lower_deterministic(wrapper(op), p))((0.5, *parents)))


def test_unbounded_truncation_needs_no_cdf_or_quantile_for_sample_and_density():
    # A bijection with no sign has neither CDF nor Quantile support.
    bijector = ir.Bijector(ir.Exp(), ir.Log(), ir.Composite(2, [ir.Identity()], [[0]]))
    base = ir.Bijected(ir.Normal(), bijector)
    op = ir.Truncated(base, lower=False, upper=False)
    actual, expected = lower_random(op, (0.3, 0.8)), lower_random(base, (0.3, 0.8))
    key = jax.random.key(0)
    assert actual.sample(key) == expected.sample(key)
    assert actual.log_prob(1.2) == expected.log_prob(1.2)
    for wrapper in [ir.CDF, ir.Quantile]:
        with pytest.raises(NotImplementedError, match="without sign"):
            lower_deterministic(wrapper(op), (0.5, 0.3, 0.8))


@pytest.mark.parametrize("lower,upper", [(True, False), (False, True), (True, True)])
def test_cdf_quantile_parameter_gradients_against_manual_jax(lower, upper):
    op = ir.Truncated(ir.Normal(), lower=lower, upper=upper)

    def actual(args):
        lo, hi, loc, scale = args
        parents = ((lo,) if lower else ()) + ((hi,) if upper else ()) + (loc, scale)
        return jnp.stack(
            (
                lower_deterministic(ir.CDF(op), (0.5, *parents)),
                lower_deterministic(ir.Quantile(op), (0.6, *parents)),
            )
        )

    def reference(args):
        lo, hi, loc, scale = args
        a = jax.scipy.special.ndtr((lo - loc) / scale) if lower else 0.0
        b = jax.scipy.special.ndtr((hi - loc) / scale) if upper else 1.0
        return jnp.stack(
            (
                (jax.scipy.special.ndtr((0.5 - loc) / scale) - a) / (b - a),
                loc + scale * jax.scipy.special.ndtri(a + 0.6 * (b - a)),
            )
        )

    args = jnp.array([-0.5, 1.5, 0.3, 0.8])
    for differentiate in [lambda f: f, jax.jacrev, jax.hessian]:
        np.testing.assert_allclose(
            jax.jit(differentiate(actual))(args),
            differentiate(reference)(args),
            rtol=2e-5,
            atol=2e-6,
        )


def test_truncated_quantile_invalid_probabilities_and_finite_endpoints():
    op = ir.Truncated(ir.Normal())
    evaluate = lambda p: lower_deterministic(ir.Quantile(op), (p, -0.5, 1.5, 0.3, 0.8))
    np.testing.assert_allclose(
        jax.jit(jax.vmap(evaluate))(jnp.array([0.0, 1.0])), [-0.5, 1.5], atol=1e-6
    )
    assert np.all(np.isnan(jax.jit(jax.vmap(evaluate))(jnp.array([-0.1, 1.1, jnp.nan]))))


def test_cdf_and_quantile_of_truncated_composite_inside_scan_and_vmap():
    # carry[t] = Quantile(Truncated(Normal(carry[t-1], exp(log_scale))))(p[t])
    normal = ir.Composite(2, [ir.Exp(), ir.Normal()], [[1], [0, 2]])
    quantile = ir.Quantile(ir.Truncated(normal))
    step = ir.Composite(
        3, [ir.Constant(-2.0), ir.Constant(2.0), quantile], [[], [], [1, 3, 4, 0, 2]]
    )
    path = ir.Scan(step, 3, (0, None))
    probabilities = jnp.array([0.2, 0.6, 0.7])
    values = lower_deterministic(path, (0.0, probabilities, -0.3))
    expected = []
    previous = 0.0
    scale = np.exp(-0.3)
    for p in probabilities:
        previous = stats.truncnorm.ppf(
            p, (-2 - previous) / scale, (2 - previous) / scale, loc=previous, scale=scale
        )
        expected.append(previous)
    np.testing.assert_allclose(values, expected, atol=1e-6)

    # The joint CDF of independent truncated draws is still a product.
    mapped = ir.CDF(ir.VMap(ir.Truncated(normal), (None, None, 0, None)))
    actual = lower_deterministic(mapped, (values, -2.0, 2.0, jnp.zeros(3), -0.3))
    expected_cdf = stats.truncnorm.cdf(values, -2 / scale, 2 / scale, scale=scale).prod()
    np.testing.assert_allclose(actual, expected_cdf, rtol=3e-6)


@pytest.mark.parametrize("lower,upper", [(True, False), (False, True), (True, True)])
def test_decreasing_bijection_truncation_cdf_quantile_and_sampling(lower, upper):
    forward = ir.Composite(1, [ir.Exp(), ir.Constant(-1.0), ir.Mul()], [[0], [], [1, 2]])
    inverse = ir.Composite(1, [ir.Constant(-1.0), ir.Mul(), ir.Log()], [[], [0, 1], [2]])
    bijector = ir.Bijector(
        forward, inverse, ir.Composite(2, [ir.Identity()], [[0]]), sign=ir.Constant(-1)
    )
    op = ir.Truncated(ir.Bijected(ir.Normal(), bijector), lower=lower, upper=upper)
    parents = ((-3.0,) if lower else ()) + ((-0.4,) if upper else ()) + (0.3, 0.8)
    reference = stats.lognorm(0.8, scale=np.exp(0.3))
    a, b = reference.sf(3.0) if lower else 0.0, reference.sf(0.4) if upper else 1.0
    probabilities = jnp.array([0.1, 0.3, 0.6, 0.9])
    values = jax.jit(jax.vmap(lambda p: lower_deterministic(ir.Quantile(op), (p, *parents))))(
        probabilities
    )
    np.testing.assert_allclose(values, -reference.isf(a + probabilities * (b - a)), rtol=3e-6)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(lambda x: lower_deterministic(ir.CDF(op), (x, *parents))))(values),
        probabilities,
        atol=3e-7,
    )
    distribution = lower_random(op, parents)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(distribution.log_prob))(values),
        reference.logpdf(-values) - np.log(b - a),
        atol=3e-6,
    )
    samples = jax.jit(jax.vmap(distribution.sample))(jax.random.split(jax.random.key(71), 2048))
    assert stats.kstest((reference.sf(-samples) - a) / (b - a), "uniform").statistic < 0.04


def test_runtime_validity_changes_under_one_jitted_vmap():
    op = ir.Truncated(ir.Uniform())

    def evaluate(lo, hi):
        parents = (lo, hi, 0.0, 1.0)
        distribution = lower_random(op, parents)
        return jnp.stack(
            (
                distribution.sample(jax.random.key(0)),
                distribution.log_prob(0.5),
                lower_deterministic(ir.CDF(op), (0.5, *parents)),
                lower_deterministic(ir.Quantile(op), (0.5, *parents)),
            )
        )

    result = jax.jit(jax.vmap(evaluate))(
        jnp.array([0.2, 0.0, 0.2, 0.8, 0.3]),
        jnp.array([0.8, 0.8, jnp.inf, 0.2, 0.7]),
    )
    assert np.all(np.isnan(result[1:4]))
    for index, lo, hi in [(0, 0.2, 0.8), (4, 0.3, 0.7)]:
        assert lo < result[index, 0] < hi
        np.testing.assert_allclose(result[index, 1:], [-np.log(hi - lo), 0.5, 0.5], atol=1e-6)
