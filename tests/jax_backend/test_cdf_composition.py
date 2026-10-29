"""Check CDFs as ingredients of differentiated, recurrent, and probabilistic computations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from pangolin import ir
from pangolin.jax_backend import ancestor_log_prob, ancestor_sample_graph, lower_deterministic


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("values", [[0.7, 1.5, 2.3], [0.2, 1.5, 2.3]])
def test_vjp_of_joint_cdf_of_vmapped_bijected_normals(jit: bool, values: list[float]):
    bijector = ir.Bijector(
        ir.Exp(),
        ir.Log(),
        ir.Composite(2, (ir.Identity(),), ((0,),)),
        sign=ir.Constant(1),
    )
    distribution = ir.VMap(ir.Bijected(ir.Normal(), bijector), (0, None))
    derivative = ir.VJP(ir.CDF(distribution), 0)
    locations = jnp.array([-0.2, 0.3, 0.6])
    scale, weight = 0.8, -1.7

    def evaluate(value: jax.Array) -> jax.Array:
        return lower_deterministic(derivative, (weight, value, locations, scale))

    actual = (jax.jit(evaluate) if jit else evaluate)(jnp.array(values))
    cdfs = stats.lognorm.cdf(values, scale, scale=np.exp(np.asarray(locations)))
    densities = stats.lognorm.pdf(values, scale, scale=np.exp(np.asarray(locations)))
    expected = np.array(
        [weight * densities[i] * np.prod(np.delete(cdfs, i)) for i in range(len(values))]
    )
    assert actual.shape == (3,)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize("jit", [False, True])
def test_deterministic_scan_with_cdf_in_its_recurrence(jit: bool):
    # carry[t] = Phi(carry[t-1] + inputs[t])
    step = ir.Composite(
        2,
        (ir.Add(), ir.Constant(0.0), ir.Constant(1.0), ir.CDF(ir.Normal())),
        ((0, 1), (), (), (2, 3, 4)),
    )
    inputs = np.array([0.2, -0.7, 1.1, -0.3])
    op = ir.Scan(step, len(inputs), (0,))

    def evaluate(start: jax.Array, increments: jax.Array) -> jax.Array:
        return lower_deterministic(op, (start, increments))

    actual = (jax.jit(evaluate) if jit else evaluate)(jnp.asarray(-0.4), jnp.asarray(inputs))
    expected = []
    carry = -0.4
    for increment in inputs:
        carry = stats.norm.cdf(carry + increment)
        expected.append(carry)
    assert not op.random
    np.testing.assert_allclose(actual, expected, rtol=2e-6)


@pytest.mark.parametrize("threshold", [-1.0, 0.4, 1.2])
@pytest.mark.parametrize("outcome", [0.0, 1.0])
def test_cdf_parameterizes_a_downstream_bernoulli(threshold: float, outcome: float):
    value = ir.RV(ir.Constant(threshold))
    location = ir.RV(ir.Constant(0.3))
    scale = ir.RV(ir.Constant(0.8))
    probability = ir.RV(ir.CDF(ir.Normal()), value, location, scale)
    draw = ir.RV(ir.Bernoulli(), probability)

    p = stats.norm.cdf(threshold, 0.3, 0.8)
    expected = np.log(p) if outcome == 1 else np.log1p(-p)
    log_prob = ancestor_log_prob(draw)
    actual = jax.jit(log_prob)(jnp.asarray(outcome))
    np.testing.assert_allclose(actual, expected, rtol=2e-6)

    sampled = ancestor_sample_graph(draw, jax.random.key(42))
    assert isinstance(sampled, jax.Array)
    assert sampled.shape == ()
    assert sampled == 0 or sampled == 1

    # The same model can be encapsulated in a Composite and used inside LogProb.
    model = ir.Composite(3, (ir.CDF(ir.Normal()), ir.Bernoulli()), ((0, 1, 2), (3,)))
    composite_log_prob = lower_deterministic(ir.LogProb(model), (outcome, threshold, 0.3, 0.8))
    np.testing.assert_allclose(composite_log_prob, expected, rtol=2e-6)
