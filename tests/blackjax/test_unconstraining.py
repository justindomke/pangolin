import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from pangolin import blackjax, functional, interface as pi, ir


def repeat_initial(log_prob, key, initial_state, num_samples):
    """Exercise the target without running a numerical inference algorithm."""
    assert jnp.isfinite(log_prob(initial_state))
    return [jnp.broadcast_to(value, (num_samples, *value.shape)) for value in initial_state]


def unexpected(*args, **kwargs):
    raise AssertionError("This callback must not run")


def test_prior_only_skips_rewriting_and_inference_even_for_discrete_outputs(monkeypatch):
    monkeypatch.setattr(functional, "transform_for", unexpected)
    x = pi.exponential(1.0)
    y = pi.poisson(x)
    xs, ys = blackjax.sample_flat([x, y], [], [], run_inf=unexpected, num_samples=8)
    assert xs.shape == ys.shape == (8,)
    assert jnp.all(xs > 0)
    assert jnp.all(ys >= 0)
    assert jnp.all(ys == jnp.floor(ys))


def test_only_active_random_nodes_are_rewritten_and_outputs_share_reconstructions(monkeypatch):
    default = functional.transform_for
    x = pi.exponential(1.0)
    shared = x + x
    observed = pi.normal(0.0, shared)
    predictive = pi.poisson(shared)
    seen = []

    def policy(op, parents):
        seen.append((op, parents))
        return default(op, parents)

    monkeypatch.setattr(functional, "transform_for", policy)

    def driver(log_prob, key, initial_state, num_samples):
        assert len(initial_state) == 1
        u = jnp.log(2.0)
        expected = dist.Exponential(1.0).log_prob(2.0) + u
        expected += dist.Normal(0.0, 4.0).log_prob(0.5)
        assert jnp.allclose(log_prob([u]), expected)
        return [jnp.full((num_samples,), u)]

    xs, copies, shareds, predictions, observations = blackjax.sample_flat(
        [x, x, shared, predictive, observed],
        [observed],
        [0.5],
        run_inf=driver,
        num_samples=8,
    )
    assert len(seen) == 1
    assert seen[0][0] == x.op
    assert seen[0][1][0] is not x.parents[0]
    assert seen[0][1][0].interface is x.interface
    assert jnp.all(xs == 2.0)
    assert jnp.array_equal(copies, xs)
    assert jnp.array_equal(shareds, 2 * xs)
    assert jnp.all(observations == 0.5)
    assert predictions.shape == (8,)


def test_observed_positive_and_discrete_nodes_are_not_passed_to_policy(monkeypatch):
    default = functional.transform_for
    x = pi.gamma(2.0, 1.0)
    y = pi.exponential(x)
    z = pi.poisson(x)
    seen = []

    def policy(op, parents):
        seen.append(op)
        return default(op, parents)

    monkeypatch.setattr(functional, "transform_for", policy)

    xs, ys, zs = blackjax.sample_flat(
        [x, y, z],
        [y, z],
        [0.5, 3],
        run_inf=repeat_initial,
        num_samples=4,
    )
    assert seen == [ir.Gamma()]
    assert jnp.all(xs > 0)
    assert jnp.all(ys == 0.5)
    assert jnp.all(zs == 3)


def test_deferred_false_rewrites_all_unobserved_random_nodes(monkeypatch):
    default = functional.transform_for
    x = pi.exponential(1.0)
    y = pi.gamma(2.0, x)
    seen = []

    def policy(op, parents):
        seen.append(op)
        return default(op, parents)

    monkeypatch.setattr(functional, "transform_for", policy)

    xs, ys = blackjax.sample_flat(
        [x, y],
        [],
        [],
        deferred=False,
        run_inf=repeat_initial,
        num_samples=4,
    )
    assert seen == [ir.Exponential(), ir.Gamma()]
    assert jnp.all(xs > 0)
    assert jnp.all(ys > 0)


def test_deferred_custom_continuous_op_does_not_require_an_unconstraining_rule():
    class CustomExponential(ir.Exponential):
        pass

    x = pi.normal(0.0, 1.0)
    y = pi.normal(x, 1.0)
    predictive = ir.RV(CustomExponential(), ir.RV(ir.Constant(1.0)))
    xs, predictions = blackjax.sample_flat(
        [x, predictive], [y], [0.0], run_inf=repeat_initial, num_samples=4
    )
    assert xs.shape == predictions.shape == (4,)
    assert jnp.all(predictions > 0)


@pytest.mark.parametrize("deferred", [False, True])
def test_default_policy_rejects_selected_discrete_nodes(deferred):
    x = pi.bernoulli(0.5)
    y = pi.normal(x, 1.0)
    with pytest.raises(NotImplementedError, match="discrete"):
        blackjax.sample_flat([x], [y], [0.0], deferred=deferred, num_samples=2)


def test_initialization_uses_observed_parents_and_density_keeps_their_factors():
    a = pi.normal(0.0, 1.0)
    x = pi.uniform(a, a + 1)
    y = pi.normal(x, 1.0)

    def driver(log_prob, key, initial_state, num_samples):
        u = initial_state[0]
        sigmoid = jax.nn.sigmoid(u)
        assert 0 < sigmoid < 1
        expected = dist.Normal(0.0, 1.0).log_prob(100.0)
        expected += dist.Normal(100 + sigmoid, 1.0).log_prob(100.5)
        expected += jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u)
        assert jnp.allclose(log_prob(initial_state), expected)
        return [jnp.broadcast_to(u, (num_samples,))]

    xs, observed_a = blackjax.sample_flat(
        [x, a], [a, y], [100.0, 100.5], run_inf=driver, num_samples=4
    )
    assert jnp.all((100 < xs) & (xs < 101))
    assert jnp.all(observed_a == 100)


def test_policy_gets_reconstructed_parents_not_original_rvs(monkeypatch):
    default = functional.transform_for
    x = pi.exponential(1.0)
    z = pi.uniform(x, x + 1)
    y = pi.normal(z, 1.0)

    def policy(op, parents):
        if isinstance(op, ir.Uniform):
            reconstructed_x = parents[0]
            assert reconstructed_x.op == ir.Exp()
            assert isinstance(reconstructed_x.parents[0].op, ir.Bijected)
            assert parents[1].parents[0] is reconstructed_x
        return default(op, parents)

    monkeypatch.setattr(functional, "transform_for", policy)

    blackjax.sample_flat([z], [y], [0.0], run_inf=repeat_initial, num_samples=4)


def test_simplex_coordinates_can_have_a_different_shape_from_requested_output():
    x = pi.dirichlet([2.0, 3.0, 4.0])
    y = pi.categorical(x)

    def driver(log_prob, key, initial_state, num_samples):
        assert len(initial_state) == 1
        assert initial_state[0].shape == (2,)
        return repeat_initial(log_prob, key, initial_state, num_samples)

    xs = blackjax.sample_flat([x], [y], [1], run_inf=driver, num_samples=4)[0]
    assert xs.shape == (4, 3)
    assert jnp.all(xs > 0)
    assert jnp.allclose(xs.sum(axis=1), 1)


def test_invalid_observations_are_rejected():
    x = pi.normal(0.0, 1.0)
    with pytest.raises(ValueError, match="length"):
        blackjax.sample_flat([x], [x], [], num_samples=2)
    with pytest.raises(ValueError, match="non-random"):
        blackjax.sample_flat([x], [x + 1], [0.0], num_samples=2)


def test_actual_nuts_on_a_constrained_gamma_poisson_posterior():
    x = pi.gamma(2.0, 1.0)
    y = pi.poisson(x)
    previous_random_state = np.random.get_state()
    try:
        np.random.seed(45)
        xs = blackjax.sample_flat([x], [y], [3], num_samples=1000)[0]
    finally:
        np.random.set_state(previous_random_state)
    assert jnp.all(xs > 0)
    assert jnp.allclose(xs.mean(), 2.5, atol=0.3)


def test_calculate_preserves_output_pytrees():
    x = pi.exponential(1.0)
    y = pi.normal(x, 1.0)
    calc = blackjax.blackjax_calculate(repeat_initial, num_samples=4)
    result = calc.sample({"x": x, "twice": (x + x,)}, y, 0.0)
    assert result["x"].shape == (4,)
    assert jnp.allclose(result["twice"][0], 2 * result["x"])
