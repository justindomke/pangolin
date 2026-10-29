import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from jax.typing import ArrayLike

from pangolin import ir
from pangolin.jax_backend import ancestor_sample_graph, lower, require_array


@pytest.mark.parametrize("value", [-2.0, 0.0, 1.5])
def test_normal_log_prob(value: float):
    actual = lower(ir.LogProb(ir.Normal()), (value, 0.3, 0.8))
    expected = -0.5 * ((value - 0.3) / 0.8) ** 2 - jnp.log(0.8) - 0.5 * jnp.log(2 * jnp.pi)

    assert isinstance(actual, jax.Array)
    assert actual.shape == ()
    assert jnp.allclose(actual, expected)


@pytest.mark.parametrize("value", [0, 1])
def test_bernoulli_log_mass(value: int):
    actual = require_array(lower(ir.LogProb(ir.Bernoulli()), (value, 0.2)))
    expected = jnp.log(0.2 if value else 0.8)

    assert actual.shape == ()
    assert jnp.allclose(actual, expected)


def test_categorical_log_mass():
    probabilities = jnp.array([0.2, 0.3, 0.5])
    actual = require_array(lower(ir.LogProb(ir.Categorical()), (1, probabilities)))

    assert jnp.allclose(actual, jnp.log(0.3))


def test_multivariate_normal_joint_log_prob():
    value = jnp.array([0.4, -0.2])
    location = jnp.array([0.1, 0.3])
    covariance = jnp.array([[1.0, 0.4], [0.4, 2.0]])
    actual = require_array(lower(ir.LogProb(ir.MultiNormal()), (value, location, covariance)))
    expected = dist.MultivariateNormal(location, covariance).log_prob(value)

    assert actual.shape == ()
    assert jnp.allclose(actual, expected)


def test_composite_log_prob_evaluates_parameter_expressions():
    base = ir.Composite(2, (ir.Exp(), ir.Normal()), ((0,), (2, 1)))
    actual = require_array(lower(ir.LogProb(base), (0.4, 0.3, 0.8)))
    expected = dist.Normal(jnp.exp(0.3), 0.8).log_prob(0.4)

    assert jnp.allclose(actual, expected)


def test_vmap_of_log_prob_differs_from_joint_log_prob():
    values = jnp.array([0.2, -0.5, 1.0])
    locations = jnp.array([0.0, 0.1, 0.2])
    individual = ir.VMap(ir.LogProb(ir.Normal()), (0, 0, None))
    joint = ir.LogProb(ir.VMap(ir.Normal(), (0, None)))
    terms = dist.Normal(locations, 0.8).log_prob(values)

    individual_value = require_array(lower(individual, (values, locations, 0.8)))
    joint_value = require_array(lower(joint, (values, locations, 0.8)))

    assert individual_value.shape == (3,)
    assert joint_value.shape == ()
    assert jnp.allclose(individual_value, terms)
    assert jnp.allclose(joint_value, terms.sum())


def test_scan_log_prob_uses_the_supplied_trajectory_as_carry():
    values = jnp.array([0.2, -0.5, 1.0])
    scales = jnp.array([0.8, 0.6, 0.9])
    base = ir.Scan(ir.Normal(), 3, (0,))
    actual = require_array(lower(ir.LogProb(base), (values, 0.3, scales)))
    previous = jnp.concatenate((jnp.array([0.3]), values[:-1]))
    expected = dist.Normal(previous, scales).log_prob(values).sum()

    assert jnp.allclose(actual, expected)


def test_nested_scan_and_vmap_log_prob():
    start = jnp.array([0.1, -0.2])
    values = jnp.array([[0.2, 0.4], [-0.3, 0.1], [0.5, 0.7]])
    base = ir.Scan(ir.VMap(ir.Normal(), (0, None)), 3, (None,))
    actual = require_array(lower(ir.LogProb(base), (values, start, 0.8)))
    previous = jnp.concatenate((start[None, :], values[:-1]), axis=0)
    expected = dist.Normal(previous, 0.8).log_prob(values).sum()

    assert actual.shape == ()
    assert jnp.allclose(actual, expected)


def test_bijected_log_prob_includes_jacobian_correction():
    bijector = ir.Bijector(ir.Exp(), ir.Log(), ir.Composite(2, (ir.Identity(),), ((0,),)))
    base = ir.Bijected(ir.Normal(), bijector)
    actual = require_array(lower(ir.LogProb(base), (1.7, 0.3, 0.8)))

    assert jnp.allclose(actual, dist.LogNormal(0.3, 0.8).log_prob(1.7))


def test_log_prob_can_be_an_intermediate_deterministic_op():
    op = ir.Composite(3, (ir.LogProb(ir.Normal()), ir.Exp()), ((0, 1, 2), (3,)))
    actual = require_array(lower(op, (0.4, 0.3, 0.8)))
    expected = jnp.exp(dist.Normal(0.3, 0.8).log_prob(0.4))

    assert jnp.allclose(actual, expected)


def test_log_prob_is_jittable_and_differentiable_in_all_parents():
    def log_prob(value, location, scale):
        return require_array(lower(ir.LogProb(ir.Normal()), (value, location, scale)))

    value, location, scale = 0.4, 0.3, 0.8
    actual = jax.jit(log_prob)(value, location, scale)
    gradients = jax.jit(jax.grad(log_prob, argnums=(0, 1, 2)))(value, location, scale)
    expected_gradients = jnp.array(
        [
            -(value - location) / scale**2,
            (value - location) / scale**2,
            -1 / scale + (value - location) ** 2 / scale**3,
        ]
    )

    assert jnp.allclose(actual, dist.Normal(location, scale).log_prob(value))
    assert jnp.allclose(jnp.array(gradients), expected_gradients)
    assert jnp.allclose(jax.grad(jax.grad(log_prob))(value, location, scale), -1 / scale**2)


def test_graph_log_prob_holds_random_parameters_fixed_without_sampling():
    zero = ir.RV(ir.Constant(0.0))
    one = ir.RV(ir.Constant(1.0))
    location = ir.RV(ir.Normal(), zero, one)
    value = ir.RV(ir.Normal(), location, one)
    logp = ir.RV(ir.LogProb(value.op), value, *value.parents)

    actual = ancestor_sample_graph(
        logp, None, boundary={value: jnp.array(0.4), location: jnp.array(0.3)}
    )

    assert isinstance(actual, jax.Array)
    assert jnp.allclose(actual, dist.Normal(0.3, 1.0).log_prob(0.4))


def test_sampling_a_value_and_its_log_prob_reuses_the_same_value():
    location = ir.RV(ir.Constant(0.3))
    scale = ir.RV(ir.Constant(0.8))
    value = ir.RV(ir.Normal(), location, scale)
    logp = ir.RV(ir.LogProb(value.op), value, *value.parents)

    sampled = ancestor_sample_graph([value, logp], jax.random.key(0))

    assert isinstance(sampled, list)
    sampled_value, sampled_logp = sampled
    assert isinstance(sampled_value, jax.Array)
    assert isinstance(sampled_logp, jax.Array)
    assert jnp.allclose(sampled_logp, dist.Normal(0.3, 0.8).log_prob(sampled_value))


@pytest.mark.parametrize("value", [jnp.ones(1), jnp.ones(3)])
def test_log_prob_rejects_invalid_value_shapes(value: ArrayLike):
    with pytest.raises(ValueError, match="value shape"):
        lower(ir.LogProb(ir.Normal()), (value, 0.0, 1.0))


def test_log_prob_rejects_invalid_distribution_parameter_shapes():
    with pytest.raises(ValueError):
        lower(ir.LogProb(ir.MultiNormal()), (jnp.zeros(3), jnp.zeros(3), jnp.eye(2)))


def test_log_prob_preserves_unsupported_base_op_error():
    class UnsupportedDistribution(ir.Op):
        random: bool = True
        """True"""
        discrete: bool = False
        """False"""

        def get_shape(self) -> ir.Shape:
            return ()

    with pytest.raises(NotImplementedError, match="No JAX lowering.*UnsupportedDistribution"):
        lower(ir.LogProb(UnsupportedDistribution()), (0.0,))
