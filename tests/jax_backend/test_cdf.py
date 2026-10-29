import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike
from scipy import stats

from pangolin import ir
from pangolin.jax_backend import ancestor_sample_graph, lower, require_array


@pytest.mark.parametrize(
    ("base", "parents", "value", "expected"),
    [
        (ir.Normal(), (0.3, 0.8), 0.4, stats.norm.cdf(0.4, 0.3, 0.8)),
        (ir.NormalPrec(), (0.3, 4.0), 0.4, stats.norm.cdf(0.4, 0.3, 0.5)),
        (ir.Beta(), (2.0, 3.0), 0.4, stats.beta.cdf(0.4, 2.0, 3.0)),
        (ir.Cauchy(), (0.3, 0.8), 0.4, stats.cauchy.cdf(0.4, 0.3, 0.8)),
        (ir.Exponential(), (2.0,), 0.4, stats.expon.cdf(0.4, scale=0.5)),
        (ir.Gamma(), (2.0, 3.0), 0.4, stats.gamma.cdf(0.4, 2.0, scale=1 / 3)),
        (ir.Lognormal(), (0.3, 0.8), 0.4, stats.lognorm.cdf(0.4, 0.8, scale=np.exp(0.3))),
        (ir.Poisson(), (2.0,), 1.5, stats.poisson.cdf(1.5, 2.0)),
        (ir.StudentT(), (5.0, 0.3, 0.8), 0.4, stats.t.cdf(0.4, 5.0, 0.3, 0.8)),
        (ir.Uniform(), (-1.0, 2.0), 0.4, stats.uniform.cdf(0.4, -1.0, 3.0)),
    ],
)
def test_primitive_cdfs_against_scipy(
    base: ir.Op, parents: tuple[ArrayLike, ...], value: float, expected: float
):
    actual = require_array(lower(ir.CDF(base), (value, *parents)))
    assert actual.shape == ()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize(
    ("base", "parents", "lo", "hi"),
    [
        (ir.Normal(), (0.3, 0.8), -np.inf, np.inf),
        (ir.NormalPrec(), (0.3, 4.0), -np.inf, np.inf),
        (ir.Beta(), (2.0, 3.0), 0.0, 1.0),
        (ir.Cauchy(), (0.3, 0.8), -np.inf, np.inf),
        (ir.Exponential(), (2.0,), 0.0, np.inf),
        (ir.Gamma(), (2.0, 3.0), 0.0, np.inf),
        (ir.Lognormal(), (0.3, 0.8), 0.0, np.inf),
        (ir.Poisson(), (2.0,), -0.1, np.inf),
        (ir.StudentT(), (5.0, 0.3, 0.8), -np.inf, np.inf),
        (ir.Uniform(), (-1.0, 2.0), -1.0, 2.0),
    ],
)
def test_support_boundaries_and_nan(
    base: ir.Op, parents: tuple[ArrayLike, ...], lo: float, hi: float
):
    evaluate = jax.jit(lambda x: require_array(lower(ir.CDF(base), (x, *parents))))
    for x in [-np.inf, lo - 1, lo]:
        assert evaluate(x) == 0.0
    for x in [hi, hi + 1, np.inf]:
        assert evaluate(x) == 1.0
    assert jnp.isnan(evaluate(np.nan))


@pytest.mark.parametrize("value", [-1.0, -0.1, 0.0, 0.5, 1.0, 1.9, 2.0, 10.0])
def test_poisson_includes_mass_at_threshold_and_floors_fractional_values(value: float):
    actual = require_array(lower(ir.CDF(ir.Poisson()), (value, 2.0)))
    np.testing.assert_allclose(actual, stats.poisson.cdf(value, 2.0), rtol=2e-6)


def test_composite_evaluates_intermediates_with_reordered_and_repeated_parents():
    base = ir.Composite(2, (ir.Mul(), ir.Exp(), ir.Normal()), ((1, 1), (2,), (0, 3)))
    actual = require_array(lower(ir.CDF(base), (0.7, 0.3, 0.8)))
    np.testing.assert_allclose(actual, stats.norm.cdf(0.7, 0.3, np.exp(0.8**2)), rtol=1e-6)


def test_zero_input_composite():
    base = ir.Composite(0, (ir.Constant(0.0), ir.Constant(1.0), ir.Normal()), ((), (), (0, 1)))
    assert require_array(lower(ir.CDF(base), (0.0,))) == 0.5


def test_single_op_composite():
    base = ir.Composite(2, (ir.Normal(),), ((1, 0),))
    np.testing.assert_allclose(
        require_array(lower(ir.CDF(base), (0.7, 0.8, 0.3))),
        stats.norm.cdf(0.7, 0.3, 0.8),
        rtol=1e-6,
    )


def test_vmap_joint_cdf_is_product_not_vector_of_marginal_cdfs():
    values = jnp.array([0.2, -0.5, 1.0])
    locations = jnp.array([0.0, 0.1, 0.2])
    individual = ir.VMap(ir.CDF(ir.Normal()), (0, 0, None))
    joint = ir.CDF(ir.VMap(ir.Normal(), (0, None)))
    probabilities = stats.norm.cdf(values, locations, 0.8)
    individual_value = require_array(lower(individual, (values, locations, 0.8)))
    joint_value = require_array(lower(joint, (values, locations, 0.8)))
    assert individual_value.shape == (3,)
    assert joint_value.shape == ()
    np.testing.assert_allclose(individual_value, probabilities, rtol=1e-6)
    np.testing.assert_allclose(joint_value, probabilities.prod(), rtol=1e-6)


def test_vmap_with_only_unmapped_parameters():
    base = ir.VMap(ir.Normal(), (None, None), 3)
    assert require_array(lower(ir.CDF(base), (jnp.zeros(3), 0.0, 1.0))) == 0.125


def test_nested_composite_and_nonleading_axis_vmap():
    scalar = ir.Composite(2, (ir.Exp(), ir.Normal()), ((1,), (0, 2)))
    base = ir.Composite(2, (ir.VMap(ir.VMap(scalar, (0, None)), (1, None)),), ((0, 1),))
    locations = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    values = jnp.ones((3, 2))
    actual = jax.jit(lambda loc: require_array(lower(ir.CDF(base), (values, loc, 0.2))))(locations)
    expected = stats.norm.cdf(values, locations.T, np.exp(0.2)).prod()
    np.testing.assert_allclose(actual, expected, rtol=2e-6)


def test_cdf_derivative_is_density_via_vjp():
    op = ir.VJP(ir.CDF(ir.Normal()), 0)
    actual = jax.jit(lambda x: require_array(lower(op, (1.0, x, 0.3, 0.8))))(0.4)
    np.testing.assert_allclose(actual, stats.norm.pdf(0.4, 0.3, 0.8), rtol=1e-6)


@pytest.mark.parametrize("value", [-1.0, 0.0, np.inf, -np.inf])
def test_out_of_support_lognormal_gradients_do_not_become_nan(value: float):
    evaluate = lambda x, loc: require_array(lower(ir.CDF(ir.Lognormal()), (x, loc, 0.8)))
    dx, dloc = jax.jit(jax.grad(evaluate, (0, 1)))(value, 0.3)
    assert dx == 0.0
    assert dloc == 0.0


@pytest.mark.parametrize(
    ("base", "parents"),
    [
        (ir.Bernoulli(), (0.3,)),
        (ir.BernoulliLogit(), (0.3,)),
        (ir.Binomial(), (5, 0.3)),
        (ir.BetaBinomial(), (5, 2.0, 3.0)),
        (ir.Categorical(), (jnp.array([0.2, 0.8]),)),
        (ir.MultiNormal(), (jnp.zeros(2), jnp.eye(2))),
        (ir.Dirichlet(), (jnp.ones(3),)),
        (ir.Multinomial(), (5, jnp.array([0.2, 0.8]))),
        (ir.Wishart(), (5, jnp.eye(2))),
        (ir.Scan(ir.Normal(), 3, (None,)), (0.0, 1.0)),
        (
            ir.Bijected(
                ir.Normal(),
                ir.Bijector(
                    ir.Exp(),
                    ir.Log(),
                    ir.Composite(2, (ir.Identity(),), ((0,),)),
                ),
            ),
            (0.0, 1.0),
        ),
    ],
)
def test_unsupported_bases_raise_explicitly(base: ir.Op, parents: tuple[ArrayLike, ...]):
    value = jnp.zeros(base.get_shape(*(jnp.shape(p) for p in parents)))
    with pytest.raises(
        NotImplementedError, match=f"CDF is not implemented for {type(base).__name__}"
    ):
        lower(ir.CDF(base), (value, *parents))


def test_unsupported_nested_scan_does_not_use_product_of_step_cdfs():
    base = ir.Composite(2, (ir.Scan(ir.Normal(), 3, (None,)),), ((0, 1),))
    with pytest.raises(NotImplementedError, match="Scan"):
        lower(ir.CDF(base), (jnp.zeros(3), 0.0, 1.0))


def test_shape_mismatch_rejected_before_numerical_evaluation():
    with pytest.raises(ValueError, match="value shape"):
        lower(ir.CDF(ir.Normal()), (jnp.zeros(3), 0.0, 1.0))


def test_graph_cdf_uses_given_parent_values_without_marginalization():
    rate = ir.RV(ir.Constant(2.0))
    location = ir.RV(ir.Exponential(), rate)
    value = ir.RV(ir.Constant(0.4))
    scale = ir.RV(ir.Constant(0.8))
    probability = ir.RV(ir.CDF(ir.Normal()), value, location, scale)
    actual = ancestor_sample_graph(probability, None, {location: jnp.asarray(0.3)})
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, stats.norm.cdf(0.4, 0.3, 0.8), rtol=1e-6)
