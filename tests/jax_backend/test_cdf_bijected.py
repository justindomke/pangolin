import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from pangolin import ir
from pangolin.jax_backend import lower, require_array, require_distribution


def exp_bijector(decreasing: bool = False) -> ir.Bijector:
    """Construct an exponential or negative exponential bijector directly in the IR."""
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


def sigmoid_bijector() -> ir.Bijector:
    """Construct a bijector with two finite output endpoints."""
    forward = ir.Composite(
        1,
        (ir.Constant(-1.0), ir.Mul(), ir.Exp(), ir.Constant(1.0), ir.Add(), ir.Div()),
        ((), (0, 1), (2,), (), (3, 4), (4, 5)),
    )
    inverse = ir.Composite(
        1, (ir.Constant(1.0), ir.Sub(), ir.Div(), ir.Log()), ((), (1, 0), (0, 2), (3,))
    )
    ldj = ir.Composite(
        2, (ir.Constant(1.0), ir.Sub(), ir.Mul(), ir.Log()), ((), (2, 1), (1, 3), (4,))
    )
    return ir.Bijector(
        forward,
        inverse,
        ldj,
        sign=ir.Constant(1),
    )


def scaling_bijector() -> ir.Bijector:
    """Construct a scaling bijector on the reals with parameter-dependent orientation."""
    ldj = ir.Composite(3, (ir.Abs(), ir.Log()), ((2,), (3,)))
    sign = ir.Composite(1, (ir.Abs(), ir.Div()), ((0,), (0, 1)))
    return ir.Bijector(ir.Mul(), ir.Div(), ldj, 1, sign=sign)


@pytest.mark.parametrize("decreasing", [False, True])
@pytest.mark.parametrize("value", [0.1, 0.5, 2.0, 8.0, np.nan])
def test_exponential_cdf_interior(decreasing: bool, value: float):
    value = -value if decreasing else value
    op = ir.CDF(ir.Bijected(ir.Normal(), exp_bijector(decreasing)))
    actual = jax.jit(lambda y: require_array(lower(op, (y, 0.3, 0.8))))(value)
    expected = (
        stats.lognorm.sf(-value, 0.8, scale=np.exp(0.3))
        if decreasing
        else stats.lognorm.cdf(value, 0.8, scale=np.exp(0.3))
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, equal_nan=True)


@pytest.mark.parametrize("value", [0.01, 0.2, 0.8, 0.99])
def test_finite_open_output_interval(value: float):
    op = ir.CDF(ir.Bijected(ir.Normal(), sigmoid_bijector()))
    actual = require_array(lower(op, (value, 0.0, 1.0)))
    expected = (
        0.0 if value <= 0 else 1.0 if value >= 1 else stats.norm.cdf(np.log(value / (1 - value)))
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6)


@pytest.mark.parametrize("decreasing", [False, True])
def test_interior_derivative_matches_density(decreasing: bool):
    base = ir.Bijected(ir.Normal(), exp_bijector(decreasing))
    value = -1.5 if decreasing else 1.5
    derivative = jax.grad(lambda y: require_array(lower(ir.CDF(base), (y, 0.3, 0.8))))(value)
    density = jnp.exp(require_distribution(lower(base, (0.3, 0.8))).log_prob(value))
    np.testing.assert_allclose(derivative, density, rtol=2e-6)


def test_bounds_parameters_and_orientation_can_change_under_jit_and_vmap():
    op = ir.CDF(ir.Bijected(ir.Uniform(), scaling_bijector()))
    evaluate = jax.jit(jax.vmap(lambda y, scale: require_array(lower(op, (y, scale, 0.0, 1.0)))))
    values = jnp.array([0.5, -0.5, 3.0, -3.0])
    scales = jnp.array([2.0, -2.0, 2.0, -2.0])
    np.testing.assert_allclose(evaluate(values, scales), [0.25, 0.75, 1.0, 0.0])


def test_parameter_gradients_at_and_outside_finite_bounds():
    op = ir.CDF(ir.Bijected(ir.Uniform(), scaling_bijector()))
    gradient = jax.jit(jax.grad(lambda scale, y: require_array(lower(op, (y, scale, 0.0, 1.0)))))
    for scale, value in [(2.0, -1.0), (2.0, 2.0), (-2.0, -2.0), (-2.0, 1.0)]:
        assert gradient(scale, value) == 0.0
    np.testing.assert_allclose(gradient(2.0, 0.5), -0.125)


def test_vmap_bijected_checks_each_mapped_parameter_range():
    scalar = ir.Bijected(ir.Uniform(), scaling_bijector())
    op = ir.CDF(ir.VMap(scalar, (0, None, None)))
    value = require_array(lower(op, (jnp.array([0.5, -0.5]), jnp.array([2.0, -2.0]), 0.0, 1.0)))
    np.testing.assert_allclose(value, 0.25 * 0.75)


def test_nested_bijected_composite_and_vmap():
    inner = ir.Bijected(ir.Normal(), exp_bijector())
    base = ir.Composite(2, (ir.Bijected(inner, sigmoid_bijector()),), ((0, 1),))
    op = ir.CDF(ir.VMap(base, (None, None), 2))
    values = jnp.array([0.7, 0.8])
    expected = stats.norm.cdf(np.log(np.log(values / (1 - values)))).prod()
    np.testing.assert_allclose(require_array(lower(op, (values, 0.0, 1.0))), expected, rtol=2e-6)


def test_base_support_can_be_smaller_than_bijector_domain():
    op = ir.CDF(ir.Bijected(ir.Uniform(), exp_bijector()))
    for value, expected in [(0.5, 0.0), (1.0, 0.0), (np.exp(0.5), 0.5), (4.0, 1.0)]:
        np.testing.assert_allclose(require_array(lower(op, (value, 0.0, 1.0))), expected, atol=1e-7)


@pytest.mark.parametrize("value", [-np.inf, -1.0, 0.5, 2.0, np.inf])
def test_unbounded_output_with_finite_input_endpoint(value: float):
    bijector = ir.Bijector(
        ir.Log(),
        ir.Exp(),
        ir.Composite(2, (ir.Constant(-1.0), ir.Mul()), ((), (1, 2))),
        sign=ir.Constant(1),
    )
    op = ir.CDF(ir.Bijected(ir.Uniform(), bijector))
    expected = stats.uniform.cdf(np.exp(value), 1.0, 3.0)
    np.testing.assert_allclose(require_array(lower(op, (value, 1.0, 4.0))), expected, atol=1e-7)


def test_two_decreasing_bijectors():
    # Negating twice is increasing, but each layer independently handles its own orientation.
    negative = ir.Composite(1, (ir.Constant(-1.0), ir.Mul()), ((), (0, 1)))
    bijector = ir.Bijector(
        negative,
        negative,
        ir.Composite(2, (ir.Constant(0.0),), ((),)),
        sign=ir.Constant(-1),
    )
    base = ir.Bijected(ir.Bijected(ir.Normal(), bijector), bijector)
    actual = require_array(lower(ir.CDF(base), (0.7, 0.3, 0.8)))
    np.testing.assert_allclose(actual, stats.norm.cdf(0.7, 0.3, 0.8), rtol=2e-6)


def test_scalar_to_vector_bijector_is_not_a_scalar_application():
    bijector = ir.Bijector(
        ir.VMap(ir.Identity(), (None,), 1),
        ir.Sum(0),
        ir.Composite(2, (ir.Constant(0.0),), ((),)),
        sign=ir.Constant(1),
    )
    base = ir.Bijected(ir.Normal(), bijector)
    with pytest.raises(NotImplementedError, match="non-scalar Bijected"):
        lower(ir.CDF(base), (jnp.zeros(1), 0.0, 1.0))


def test_missing_sign_fails_through_nested_wrappers():
    no_sign = ir.Bijector(ir.Exp(), ir.Log(), ir.Composite(2, (ir.Identity(),), ((0,),)))
    inner = ir.Bijected(ir.Normal(), no_sign)
    base = ir.VMap(ir.Bijected(inner, exp_bijector()), (None, None), 2)
    with pytest.raises(NotImplementedError, match="without sign"):
        lower(ir.CDF(base), (jnp.array([-1.0, -2.0]), 0.0, 1.0))


@pytest.mark.parametrize("nested", [False, True])
def test_missing_sign_is_not_inferred_from_forward(nested: bool):
    prototype = exp_bijector()
    bijector = ir.Bijector(
        prototype.forward,
        prototype.inverse,
        prototype.log_det_jac,
    )
    base = ir.Bijected(ir.Normal(), bijector)
    if nested:
        op = ir.CDF(ir.VMap(base, (None, None), 2))
        value = jnp.array([-1.0, -2.0])
    else:
        op = ir.CDF(base)
        value = jnp.asarray(-1.0)
    with pytest.raises(NotImplementedError, match="without sign"):
        lower(op, (value, 0.0, 1.0))


def test_sign_enables_cdf_without_bounds():
    prototype = exp_bijector()
    actual = lower(ir.CDF(ir.Bijected(ir.Normal(), prototype)), (1.0, 0.0, 1.0))
    assert require_array(actual) == 0.5


def test_bijected_vmap_rejects_non_scalar_application():
    bijector = ir.Bijector(
        ir.Identity(),
        ir.Identity(),
        ir.Composite(2, (ir.Constant(0.0),), ((),)),
        sign=ir.Constant(1),
    )
    base = ir.Bijected(ir.VMap(ir.Normal(), (None, None), 2), bijector)
    with pytest.raises(NotImplementedError, match="non-scalar Bijected"):
        lower(ir.CDF(base), (jnp.zeros(2), 0.0, 1.0))


def test_sampling_and_density_ignore_sign():
    class UnlowerableSign(ir.Op):
        random = False

        def get_shape(self) -> ir.Shape:
            return ()

    prototype = exp_bijector()
    bijector = ir.Bijector(
        prototype.forward, prototype.inverse, prototype.log_det_jac, sign=UnlowerableSign()
    )
    base = require_distribution(lower(ir.Bijected(ir.Normal(), prototype), (0.0, 1.0)))
    other = require_distribution(lower(ir.Bijected(ir.Normal(), bijector), (0.0, 1.0)))
    assert base.sample(jax.random.key(0)) == other.sample(jax.random.key(0))
    assert base.log_prob(2.0) == other.log_prob(2.0)
    with pytest.raises(NotImplementedError, match="UnlowerableSign"):
        lower(ir.CDF(ir.Bijected(ir.Normal(), bijector)), (2.0, 0.0, 1.0))
