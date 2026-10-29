"""Truncated densities, inverse-CDF samples, and recursive CDF/Quantile lowering."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from pangolin import interface as pi, ir, jax_backend
from pangolin.jax_backend import lower_deterministic, lower_random


@pytest.mark.parametrize(
    "base,parents,reference,lo,hi",
    [
        (ir.Normal(), (0.3, 0.8), stats.norm(0.3, 0.8), -0.5, 1.5),
        (ir.NormalPrec(), (0.3, 4.0), stats.norm(0.3, 0.5), -0.5, 1.5),
        (ir.Cauchy(), (0.3, 0.8), stats.cauchy(0.3, 0.8), -1.0, 2.0),
        (ir.Exponential(), (2.0,), stats.expon(scale=0.5), 0.2, 2.0),
        (ir.Lognormal(), (0.3, 0.8), stats.lognorm(0.8, scale=np.exp(0.3)), 0.2, 2.0),
        (ir.Uniform(), (-1.0, 2.0), stats.uniform(-1.0, 3.0), -0.5, 1.5),
    ],
)
def test_primitives_density_and_sampling_against_scipy(base, parents, reference, lo, hi):
    distribution = lower_random(ir.Truncated(base), (lo, hi, *parents))
    values = jnp.linspace(lo, hi, 11)
    mass = reference.cdf(hi) - reference.cdf(lo)
    expected = reference.logpdf(values) - np.log(mass)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(distribution.log_prob))(values), expected, atol=2e-6
    )
    for value in [-np.inf, lo - 1, hi + 1, np.inf]:
        assert distribution.log_prob(value) == -np.inf
    assert jnp.isnan(distribution.log_prob(np.nan))

    keys = jax.random.split(jax.random.key(12), 4096)
    samples = np.asarray(jax.jit(jax.vmap(distribution.sample))(keys))
    assert np.all((lo <= samples) & (samples <= hi))
    probabilities = (reference.cdf(samples) - reference.cdf(lo)) / mass
    assert stats.kstest(probabilities, "uniform").statistic < 0.03
    np.testing.assert_array_equal(distribution.sample(keys[0]), distribution.sample(keys[0]))


@pytest.mark.parametrize("lo,hi", [(-np.inf, np.inf), (-np.inf, 0.5), (0.5, np.inf)])
def test_omitted_bounds_and_parameter_gradients(lo, hi):
    key = jax.random.key(4)

    op = ir.Truncated(ir.Normal(), lower=bool(np.isfinite(lo)), upper=bool(np.isfinite(hi)))
    bounds = tuple(v for v in (lo, hi) if np.isfinite(v))

    def sample(loc):
        return lower_random(op, (*bounds, loc, 0.8)).sample(key)

    def density(loc):
        return lower_random(op, (*bounds, loc, 0.8)).log_prob(0.5)

    expected_density = stats.truncnorm.logpdf(
        0.5, (lo - 0.3) / 0.8, (hi - 0.3) / 0.8, loc=0.3, scale=0.8
    )
    np.testing.assert_allclose(density(0.3), expected_density, atol=2e-6)
    for fun in [sample, density]:
        actual = jax.jit(jax.grad(fun))(0.3)
        expected = (fun(0.301) - fun(0.299)) / 0.002
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-4)


def test_gradients_with_respect_to_bounds_location_and_scale():
    key = jax.random.key(0)

    def evaluate(lo, hi, loc, scale):
        distribution = lower_random(ir.Truncated(ir.Normal()), (lo, hi, loc, scale))
        return distribution.sample(key) + distribution.log_prob(0.2)

    parents = [-0.5, 1.5, 0.3, 0.8]
    actual = jax.jit(jax.grad(evaluate, (0, 1, 2, 3)))(*parents)
    for i in range(4):
        plus, minus = parents.copy(), parents.copy()
        plus[i] += 0.001
        minus[i] -= 0.001
        expected = (evaluate(*plus) - evaluate(*minus)) / 0.002
        np.testing.assert_allclose(actual[i], expected, atol=3e-4, rtol=2e-3)


@pytest.mark.parametrize(
    "lo,hi", [(1.0, 1.0), (2.0, 1.0), (np.nan, 1.0), (0.0, np.nan), (np.inf, np.inf)]
)
def test_invalid_intervals_return_nan_under_jit(lo, hi):
    def evaluate(lo, hi):
        distribution = lower_random(ir.Truncated(ir.Normal()), (lo, hi, 0.0, 1.0))
        return distribution.sample(jax.random.key(0)), distribution.log_prob(0.0)

    assert np.all(np.isnan(jax.jit(evaluate)(lo, hi)))


def test_zero_mass_and_unresolved_tail_intervals_return_nan():
    for base, parents, lo, hi in [
        (ir.Uniform(), (0.0, 1.0), 2.0, 3.0),
        (ir.Exponential(), (1.0,), -2.0, -1.0),
        (ir.Normal(), (0.0, 1.0), 100.0, 101.0),
    ]:
        distribution = lower_random(ir.Truncated(base), (lo, hi, *parents))
        assert jnp.isnan(distribution.sample(jax.random.key(0)))
        assert jnp.isnan(distribution.log_prob(lo))


def test_zero_uniform_draw_preserves_normal_quantile_endpoint(monkeypatch):
    monkeypatch.setattr(jax.random, "uniform", lambda key, **kwargs: jnp.array(0.0))
    distribution = lower_random(ir.Truncated(ir.Normal(), lower=False), (2.0, 0.0, 1.0))
    assert jax.jit(distribution.sample)(jax.random.key(0)) == -jnp.inf


@pytest.mark.parametrize(
    "base,parents,reference,lo,hi,value",
    [
        (ir.Exponential(), (2.0,), stats.expon(scale=0.5), -2.0, 2.0, 0.5),
        (ir.Uniform(), (0.0, 1.0), stats.uniform(), -2.0, 2.0, 0.5),
        (ir.Lognormal(), (0.0, 1.0), stats.lognorm(1.0), -2.0, 2.0, 0.5),
    ],
)
def test_truncation_wider_than_base_support_is_invalid(base, parents, reference, lo, hi, value):
    distribution = lower_random(ir.Truncated(base), (lo, hi, *parents))
    assert jnp.isnan(distribution.log_prob(value))
    assert jnp.isnan(distribution.sample(jax.random.key(0)))


@pytest.mark.parametrize(
    "base,parents,reference",
    [
        (ir.Beta(), (2.0, 3.0), stats.beta(2.0, 3.0)),
        (ir.Gamma(), (2.0, 3.0), stats.gamma(2.0, scale=1 / 3)),
        (ir.StudentT(), (5.0, 0.0, 1.0), stats.t(5.0)),
    ],
)
def test_density_does_not_require_quantile(base, parents, reference):
    distribution = lower_random(ir.Truncated(base), (0.2, 0.8, *parents))
    expected = reference.logpdf(0.5) - np.log(reference.cdf(0.8) - reference.cdf(0.2))
    np.testing.assert_allclose(distribution.log_prob(0.5), expected, atol=4e-6)
    with pytest.raises(NotImplementedError, match="Quantile is not implemented"):
        distribution.sample(jax.random.key(0))


def exp_bijector(decreasing=False):
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
@pytest.mark.parametrize("lo,hi", [(0.2, 2.0), (0.4, 5.0)])
def test_bijected_sampling_and_density_use_base_cdf_and_quantile(decreasing, lo, hi):
    base_lo = np.log(lo) if lo > 0 else -np.inf
    base_hi = np.log(hi)
    if decreasing:
        lo, hi = -hi, -lo
    op = ir.Truncated(ir.Bijected(ir.Normal(), exp_bijector(decreasing)))
    distribution = lower_random(op, (lo, hi, 0.3, 0.8))
    base = lower_random(ir.Truncated(ir.Normal()), (base_lo, base_hi, 0.3, 0.8))
    key = jax.random.key(3)
    sign = -1.0 if decreasing else 1.0
    a = stats.norm.cdf(base_lo, 0.3, 0.8)
    b = stats.norm.cdf(base_hi, 0.3, 0.8)
    p = jax.random.uniform(
        key, minval=1 - b if decreasing else a, maxval=1 - a if decreasing else b
    )
    expected_sample = sign * np.exp(stats.norm.ppf(1 - p if decreasing else p, 0.3, 0.8))
    np.testing.assert_allclose(jax.jit(distribution.sample)(key), expected_sample, rtol=2e-6)
    value = sign * 1.5
    expected = base.log_prob(np.log(abs(value))) - np.log(abs(value))
    np.testing.assert_allclose(distribution.log_prob(value), expected, atol=2e-6)
    for outside in [0.0, -sign, lo - 1.0, hi + 1.0]:
        assert distribution.log_prob(outside) == -np.inf
    assert jax.jit(jax.grad(distribution.log_prob))(-sign) == 0.0


def test_parameter_dependent_bijector_orientation_under_vmap_and_grad():
    sign = ir.Composite(1, (ir.Abs(), ir.Div()), ((0,), (0, 1)))
    ldj = ir.Composite(3, (ir.Abs(), ir.Log()), ((2,), (3,)))
    bijector = ir.Bijector(ir.Mul(), ir.Div(), ldj, 1, sign=sign)
    op = ir.Truncated(ir.Bijected(ir.Normal(), bijector))

    def evaluate(scale):
        return lower_random(op, (-1.0, 2.0, scale, 0.3, 0.8)).log_prob(0.2)

    scales = jnp.array([-2.0, 3.0])
    expected = [
        stats.truncnorm.logpdf(
            0.2,
            (-1 - s * 0.3) / abs(s * 0.8),
            (2 - s * 0.3) / abs(s * 0.8),
            loc=s * 0.3,
            scale=abs(s * 0.8),
        )
        for s in scales
    ]
    np.testing.assert_allclose(jax.jit(jax.vmap(evaluate))(scales), expected, atol=2e-6)
    actual = jax.jit(jax.vmap(jax.grad(evaluate)))(scales)
    expected_grad = jax.vmap(lambda s: (evaluate(s + 0.001) - evaluate(s - 0.001)) / 0.002)(scales)
    np.testing.assert_allclose(actual, expected_grad, atol=3e-4)


def test_bijected_empty_or_reversed_intervals_do_not_become_valid_after_mapping():
    for decreasing in [False, True]:
        op = ir.Truncated(ir.Bijected(ir.Normal(), exp_bijector(decreasing)))
        invalid = [(2.0, 1.0), (1.0, 1.0), (np.nan, 2.0)]
        invalid += [(1.0, 2.0)] if decreasing else [(-2.0, -1.0)]
        for lo, hi in invalid:
            distribution = lower_random(op, (lo, hi, 0.0, 1.0))
            assert jnp.isnan(jax.jit(distribution.sample)(jax.random.key(0)))
            assert jnp.isnan(distribution.log_prob(0.5))


def test_bijector_range_does_not_replace_base_support():
    op = ir.Truncated(ir.Bijected(ir.Uniform(), exp_bijector()))
    valid = lower_random(op, (1.2, 2.5, 0.0, 1.0))
    np.testing.assert_allclose(
        valid.log_prob(2.0), -np.log(2.0) - np.log(np.log(2.5) - np.log(1.2)), atol=1e-6
    )
    for value in [0.5, 4.0, -1.0]:
        assert valid.log_prob(value) == -np.inf
    invalid = lower_random(op, (4.0, 10.0, 0.0, 1.0))
    assert jnp.isnan(invalid.sample(jax.random.key(0)))
    for value in [-1.0, 5.0, 11.0]:
        assert jnp.isnan(invalid.log_prob(value))


def test_finite_bijector_range_handles_bounds_and_gradients_without_invalid_inverse():
    ldj = ir.Composite(
        2,
        (ir.Constant(1.0), ir.Sub(), ir.Mul(), ir.Log()),
        ((), (2, 1), (1, 3), (4,)),
    )
    bijector = ir.Bijector(
        ir.InvLogit(),
        ir.Logit(),
        ldj,
        sign=ir.Constant(1),
    )
    op = ir.Truncated(ir.Bijected(ir.Normal(), bijector))

    def density(lo, hi):
        return lower_random(op, (lo, hi, 0.0, 1.0)).log_prob(0.5)

    lo, hi = 0.2, 0.8
    mass = stats.norm.cdf(np.log(hi / (1 - hi))) - stats.norm.cdf(np.log(lo / (1 - lo)))
    expected = stats.norm.logpdf(0.0) - np.log(0.25) - np.log(mass)
    np.testing.assert_allclose(jax.jit(density)(lo, hi), expected, atol=1e-6)
    assert np.all(np.isfinite(jax.jit(jax.grad(density, (0, 1)))(lo, hi)))
    for lo, hi in [(-2.0, 3.0), (0.0, 1.0), (-2.0, 0.0), (1.0, 3.0)]:
        assert jnp.isnan(density(lo, hi))


def test_nested_bijections_truncations_and_composites():
    normal = ir.Composite(2, (ir.Normal(),), ((0, 1),))
    base = ir.Bijected(ir.Truncated(ir.Bijected(normal, exp_bijector())), exp_bijector())
    actual = lower_random(ir.Truncated(base), (np.exp(0.5), np.exp(2.0), 0.2, 3.0, 0.3, 0.8))
    reference = lower_random(ir.Truncated(ir.Normal()), (np.log(0.5), np.log(2.0), 0.3, 0.8))
    key = jax.random.key(1)
    np.testing.assert_allclose(
        actual.sample(key), jnp.exp(jnp.exp(reference.sample(key))), rtol=1e-6
    )
    value = np.exp(1.2)
    expected = reference.log_prob(np.log(1.2)) - np.log(1.2) - 1.2
    np.testing.assert_allclose(actual.log_prob(value), expected, atol=2e-6)


def test_nested_vmap_nonleading_axes_and_invalid_batch_members():
    op = ir.VMap(ir.VMap(ir.Truncated(ir.Normal()), (0, None, 0, None)), (1, None, 1, None))
    lo = jnp.array([[-1.0, 2.0, 0.0], [0.2, -0.3, -1.0]])
    loc = jnp.array([[0.2, 0.3, 0.4], [0.1, -0.2, 0.5]])
    distribution = lower_random(op, (lo, 1.0, loc, 0.8))
    samples = jax.jit(distribution.sample)(jax.random.key(0))
    assert samples.shape == (3, 2)
    assert jnp.isnan(samples[1, 0])
    valid = np.asarray(lo.T < 1.0)
    assert np.all(np.asarray(samples)[valid] >= np.asarray(lo.T)[valid])
    assert np.all(np.asarray(samples)[valid] <= 1.0)
    assert jnp.isnan(distribution.log_prob(jnp.zeros((3, 2))))


def test_nested_truncations_and_composites_with_strictly_nested_intervals():
    normal = ir.Composite(3, (ir.Mul(), ir.Exp(), ir.Normal()), ((1, 1), (3,), (0, 4)))
    op = ir.Truncated(ir.Composite(5, (ir.Truncated(normal),), ((0, 1, 2, 3, 4),)))
    distribution = lower_random(op, (-0.5, 0.7, -1.0, 1.0, 0.3, 0.8, jnp.ones(3)))
    expected = lower_random(ir.Truncated(ir.Normal()), (-0.5, 0.7, 0.3, np.exp(0.8**2)))
    key = jax.random.key(0)
    np.testing.assert_allclose(distribution.sample(key), expected.sample(key), atol=1e-6)
    np.testing.assert_allclose(distribution.log_prob(0.2), expected.log_prob(0.2), atol=1e-6)


def test_zero_input_composite():
    base = ir.Composite(0, (ir.Constant(0.0), ir.Constant(1.0), ir.Normal()), ((), (), (0, 1)))
    actual = lower_random(ir.Truncated(base), (-1.0, 2.0))
    expected = lower_random(ir.Truncated(ir.Normal()), (-1.0, 2.0, 0.0, 1.0))
    np.testing.assert_equal(actual.log_prob(0.3), expected.log_prob(0.3))


def test_vmap_scan_and_interface_log_prob_composition():
    means = jnp.array([0.1, 0.2, 0.3])
    model = pi.truncated(pi.normal, -1.0, 2.0)
    result = model(means, 0.8)
    values = jnp.array([0.3, 0.4, 0.5])
    logp = pi.log_prob(model)(values, means, 0.8)
    expected = stats.truncnorm.logpdf(
        values, (-1 - means) / 0.8, (2 - means) / 0.8, loc=means, scale=0.8
    ).sum()
    actual = jax_backend.ancestor_sample_graph(logp, None)
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    samples = jax_backend.ancestor_sampler(result)(jax.random.key(0))
    assert isinstance(samples, jax.Array)
    assert samples.shape == (3,)
    assert np.all((samples >= -1) & (samples <= 2))

    step = ir.Composite(
        2, (ir.Constant(-1.0), ir.Constant(2.0), ir.Truncated(ir.Normal())), ((), (), (2, 3, 0, 1))
    )
    path = lower_random(ir.Scan(step, 3, (None,)), (0.0, 0.8))
    reference = stats.truncnorm.logpdf(
        values,
        (-1 - np.array([0.0, 0.3, 0.4])) / 0.8,
        (2 - np.array([0.0, 0.3, 0.4])) / 0.8,
        loc=[0.0, 0.3, 0.4],
        scale=0.8,
    ).sum()
    np.testing.assert_allclose(jax.jit(path.log_prob)(values), reference, atol=2e-6)
    samples = jax.jit(path.sample)(jax.random.key(3))
    assert samples.shape == (3,)
    assert np.all((samples >= -1) & (samples <= 2))


def test_missing_sign_and_unsupported_scalar_base():
    bijector = ir.Bijector(ir.Exp(), ir.Log(), ir.Composite(2, (ir.Identity(),), ((0,),)))
    with pytest.raises(NotImplementedError, match="without sign"):
        lower_random(ir.Truncated(ir.Bijected(ir.Normal(), bijector)), (0.0, 2.0, 0.0, 1.0))

    class Unknown(ir.Op):
        random: bool = True
        """True"""
        discrete: bool = False
        """False"""

        def get_shape(self):
            return ()

    with pytest.raises(NotImplementedError, match="Unknown"):
        lower_random(ir.Truncated(Unknown()), (0.0, 1.0))


def test_ir_still_rejects_array_valued_truncation():
    with pytest.raises(ValueError, match="scalar output"):
        lower_random(ir.Truncated(ir.MultiNormal()), (0.0, 1.0, jnp.zeros(2), jnp.eye(2)))


def test_log_prob_wrapper_uses_truncated_density():
    actual = lower_deterministic(ir.LogProb(ir.Truncated(ir.Normal())), (0.2, -1.0, 2.0, 0.0, 1.0))
    np.testing.assert_allclose(actual, stats.truncnorm.logpdf(0.2, -1.0, 2.0), atol=1e-6)


@pytest.mark.parametrize("one_sided", [False, True])
@pytest.mark.parametrize("sampling", [False, True])
def test_exponential_hessians_against_closed_form(one_sided, sampling):
    """Check mixed second derivatives of samples and densities, including the bounds."""
    key = jax.random.key(17)
    u = jax.random.uniform(key)

    def actual(args):
        lo, width, rate = args
        parents = (lo, rate) if one_sided else (lo, lo + width, rate)
        distribution = lower_random(ir.Truncated(ir.Exponential(), upper=not one_sided), parents)
        return distribution.sample(key) if sampling else distribution.log_prob(0.7)

    def reference(args):
        lo, width, rate = args
        mass = 1.0 if one_sided else -jnp.expm1(-rate * width)
        if sampling:
            return lo - jnp.log1p(-u * mass) / rate
        return jnp.log(rate) - rate * (0.7 - lo) - jnp.log(mass)

    args = jnp.array([0.2, 1.4, 1.3])
    for differentiate in [lambda f: f, jax.grad, jax.hessian]:
        np.testing.assert_allclose(
            jax.jit(differentiate(actual))(args),
            differentiate(reference)(args),
            rtol=2e-5,
            atol=2e-6,
        )


@pytest.mark.parametrize("lo,hi", [(0.3, 0.31), (2.0, 2.5), (-2.5, -2.0)])
def test_narrow_and_moderate_tail_normal_intervals(lo, hi):
    """Resolve representable small masses; this does not promise extreme-tail accuracy."""
    distribution = lower_random(ir.Truncated(ir.Normal()), (lo, hi, 0.0, 1.0))
    reference = stats.truncnorm(lo, hi)
    points, weights = np.polynomial.legendre.leggauss(64)
    values = jnp.asarray(lo + (points + 1) * (hi - lo) / 2)
    log_density = jax.jit(jax.vmap(distribution.log_prob))(values)
    np.testing.assert_allclose(
        log_density, stats.truncnorm.logpdf(values, lo, hi), atol=4e-5, rtol=0
    )
    integral = np.dot(weights, np.exp(log_density)) * (hi - lo) / 2
    np.testing.assert_allclose(integral, 1.0, atol=4e-5, rtol=0)

    samples = np.asarray(
        jax.jit(jax.vmap(distribution.sample))(jax.random.split(jax.random.key(39), 4096))
    )
    assert np.all((samples >= lo) & (samples <= hi))
    assert stats.kstest(reference.cdf(samples), "uniform").statistic < 0.03


@pytest.mark.parametrize("decreasing", [False, True])
def test_bijected_samples_and_density_against_scipy(decreasing):
    """An independent lognormal reference checks orientation and normalization together."""
    sign = -1.0 if decreasing else 1.0
    lo, hi = (-3.0, -0.4) if decreasing else (0.4, 3.0)
    distribution = lower_random(
        ir.Truncated(ir.Bijected(ir.Normal(), exp_bijector(decreasing))),
        (lo, hi, 0.3, 0.8),
    )
    reference = stats.lognorm(0.8, scale=np.exp(0.3))
    mass = reference.cdf(3.0) - reference.cdf(0.4)
    values = jnp.linspace(lo, hi, 31)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(distribution.log_prob))(values),
        stats.lognorm.logpdf(sign * np.asarray(values), 0.8, scale=np.exp(0.3)) - np.log(mass),
        atol=3e-6,
    )
    samples = np.asarray(
        jax.jit(jax.vmap(distribution.sample))(jax.random.split(jax.random.key(42), 4096))
    )
    assert np.all((samples >= lo) & (samples <= hi))
    probabilities = (reference.cdf(sign * samples) - reference.cdf(0.4)) / mass
    if decreasing:
        probabilities = 1 - probabilities
    assert stats.kstest(probabilities, "uniform").statistic < 0.03


def test_nested_truncation_gradients_cancel_inner_bounds():
    """Inner bounds cancel from the normalized law, including pathwise derivatives."""
    key = jax.random.key(8)

    def evaluate(bounds):
        outer_lo, outer_hi, inner_lo, inner_hi = bounds
        distribution = lower_random(
            ir.Truncated(ir.Truncated(ir.Uniform())),
            (outer_lo, outer_hi, inner_lo, inner_hi, -5.0, 5.0),
        )
        return jnp.stack((distribution.sample(key), distribution.log_prob(0.5)))

    def reference(bounds):
        lo, hi = bounds[0], bounds[1]
        return jnp.stack((lo + (hi - lo) * jax.random.uniform(key), -jnp.log(hi - lo)))

    bounds = jnp.array([0.0, 1.5, -1.0, 2.0])
    for differentiate in [lambda f: f, jax.jacrev, jax.hessian]:
        np.testing.assert_allclose(
            jax.jit(differentiate(evaluate))(bounds),
            differentiate(reference)(bounds),
            atol=2e-6,
        )
