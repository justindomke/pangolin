import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pangolin import ir
from pangolin.jax_backend import (
    LoweredDistribution,
    ancestor_log_prob,
    ancestor_sampler,
    fill_in,
    lower,
    lower_deterministic,
    lower_random,
    require_distribution,
    require_array,
    ancestor_sample_graph,
)


def test_lower_deterministic_op_to_array():
    lowered = lower(ir.Add(), [2.0, 3.0])
    assert isinstance(lowered, jax.Array)
    assert lowered == 5.0

    assert jnp.allclose(require_array(lower(ir.InvLogit(), [2.0])), jax.nn.sigmoid(2.0))


def test_typed_lowering_wrappers_accept_array_like_parents():
    assert lower_deterministic(ir.Add(), (2.0, 3.0)) == 5.0
    distribution = lower_random(ir.Normal(), (0.0, 1.0))
    assert distribution.log_prob(0.5) == dist.Normal(0.0, 1.0).log_prob(0.5)


def test_typed_lowering_wrappers_reject_wrong_result_kind():
    with pytest.raises(TypeError, match="Expected a lowered array"):
        lower_deterministic(ir.Normal(), (0.0, 1.0))
    with pytest.raises(TypeError, match="Expected a lowered distribution"):
        lower_random(ir.Add(), (2.0, 3.0))


def test_lower_random_op_to_distribution():
    lowered = lower(ir.Normal(), [2.0, 3.0])
    assert isinstance(lowered, LoweredDistribution)

    key = jax.random.PRNGKey(0)
    assert lowered.sample(key).shape == ()
    assert lowered.log_prob(1.0) == dist.Normal(2.0, 3.0).log_prob(1.0)


def test_lowered_distribution_normalizes_input_and_preserves_callback_outputs():
    received: list[jax.Array] = []
    log_prob_result = jnp.asarray(-2.0)
    sample_result = jnp.asarray(1.0)

    def log_prob(value: jax.Array) -> jax.Array:
        assert isinstance(value, jax.Array)
        received.append(value)
        return log_prob_result

    distribution = LoweredDistribution(lambda key: sample_result, log_prob)
    result = distribution.log_prob(3.0)
    assert received[0] == 3.0
    assert result is log_prob_result
    assert distribution.sample(jax.random.key(0)) is sample_result


def test_lowered_distribution_enforces_the_op_shape():
    lowered = lower(ir.VMap(ir.Normal(), (None, None), 2), [0.0, 1.0])
    assert isinstance(lowered, LoweredDistribution)

    with pytest.raises(AssertionError, match="shape was not as expected"):
        lowered.log_prob(jnp.zeros(3))


def test_lower_uses_base_op_implementation_for_subclasses():
    class DerivedAdd(ir.Add):
        pass

    class DerivedNormal(ir.Normal):
        pass

    assert require_array(lower(DerivedAdd(), [2.0, 3.0])) == 5.0
    assert isinstance(lower(DerivedNormal(), [0.0, 1.0]), LoweredDistribution)


def test_lower_rejects_unsupported_op():
    class Unsupported(ir.Op):
        random: bool = False
        """False"""

        @staticmethod
        def get_shape():
            return ()

    with pytest.raises(NotImplementedError, match="Unsupported"):
        lower(Unsupported(), [])


def test_composite_uses_the_same_lowering_recursively():
    deterministic = ir.Composite(1, (ir.Constant(2.0), ir.Add()), [[], [0, 1]])
    assert require_array(lower(deterministic, [3.0])) == 5.0

    random = ir.Composite(
        1,
        (ir.Constant(2.0), ir.Add(), ir.Normal()),
        [[], [0, 1], [2, 0]],
    )
    distribution = require_distribution(lower(random, [3.0]))
    assert distribution.log_prob(4.0) == dist.Normal(5.0, 3.0).log_prob(4.0)


@pytest.mark.parametrize("axis", [1, -1])
def test_scan_supports_any_valid_input_axis(axis):
    vector_add = ir.VMap(ir.Add(), (0, 0), 2)
    op = ir.Scan(vector_add, 3, (axis,))
    start = jnp.array([10.0, 20.0])
    increments = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    actual = require_array(lower(op, [start, increments]))
    expected = start + jnp.cumsum(increments, axis=1).T
    assert jnp.array_equal(actual, expected)


def test_random_vmap_infers_axis_size_from_a_mapped_parent():
    op = ir.VMap(ir.Normal(), (0, None), None)
    lowered = lower(op, [jnp.zeros(3), 1.0])
    assert isinstance(lowered, LoweredDistribution)
    assert lowered.sample(jax.random.PRNGKey(0)).shape == (3,)


def test_graph_sampling_reuses_shared_values():
    loc = ir.RV(ir.Constant(1.0))
    scale = ir.RV(ir.Constant(0.1))
    x = ir.RV(ir.Normal(), loc, scale)
    y = ir.RV(ir.Add(), x, x)

    x_value, y_value = ancestor_sample_graph([x, y], jax.random.PRNGKey(0))
    assert y_value == 2 * x_value


def test_graph_log_prob_propagates_deterministic_values():
    one = ir.RV(ir.Constant(1.0))
    two = ir.RV(ir.Add(), one, one)
    x = ir.RV(ir.Normal(), two, one)

    actual = ancestor_log_prob(x)(jnp.array(3.0))
    expected = dist.Normal(2.0, 1.0).log_prob(3.0)
    assert actual == expected


def test_empty_graph_has_zero_log_prob():
    assert ancestor_log_prob()() == 0.0


def test_ancestor_log_prob_accepts_structured_variables_and_values():
    zero = ir.RV(ir.Constant(0.0))
    one = ir.RV(ir.Constant(1.0))
    x = ir.RV(ir.Normal(), zero, one)
    y = ir.RV(ir.Normal(), x, one)

    log_prob = ancestor_log_prob({"x": x}, ys=[y])
    actual = log_prob({"x": 2.0}, ys=[3.0])
    expected = dist.Normal(0.0, 1.0).log_prob(2.0) + dist.Normal(2.0, 1.0).log_prob(3.0)

    assert actual == expected


def test_ancestor_log_prob_handles_an_empty_pytree():
    assert ancestor_log_prob()() == 0.0


def test_graph_boundary_values_must_match_rv_shapes():
    loc = ir.RV(ir.Constant(0.0))
    scale = ir.RV(ir.Constant(1.0))
    x = ir.RV(ir.VMap(ir.Normal(), (None, None), 2), loc, scale)

    with pytest.raises(ValueError, match=r"shape \(3,\), expected \(2,\)"):
        fill_in([x], [jnp.zeros(3)], [x])


def test_fill_in_samples_random_nodes_beyond_the_boundary():
    zero = ir.RV(ir.Constant(0.0))
    one = ir.RV(ir.Constant(1.0))
    x = ir.RV(ir.Normal(), zero, one)
    y = ir.RV(ir.Normal(), x, one)
    key = jax.random.PRNGKey(0)

    [actual] = fill_in([x], [2.0], [y], key)
    _, subkey = jax.random.split(key)
    expected = dist.Normal(2.0, 1.0).sample(subkey)
    assert actual == expected


def test_compiled_graph_functions_are_jittable():
    zero = ir.RV(ir.Constant(0.0))
    one = ir.RV(ir.Constant(1.0))
    x = ir.RV(ir.Normal(), zero, one)
    y = ir.RV(ir.Add(), x, one)

    sample = jax.jit(ancestor_sampler({"x": x, "y": y}))(jax.random.PRNGKey(0))
    assert jnp.allclose(sample["y"], sample["x"] + 1)

    log_prob = jax.jit(ancestor_log_prob(x))
    assert log_prob(0.0) == dist.Normal(0.0, 1.0).log_prob(0.0)
