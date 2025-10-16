import blackjax
from jax import numpy as jnp
import jax


def inference_loop(rng_key, kernel, initial_states, num_samples):
    @jax.jit
    def one_step(states, rng_key):
        states, infos = kernel(rng_key, states)
        return states, (states, infos)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)

    return (states, infos)


def run_nuts(log_prob, key, initial_state, num_samples):
    # to do hmc instead:
    # adapt = blackjax.window_adaptation(blackjax.hmc, log_prob, num_integration_steps=60)
    # kernel = blackjax.hmc(log_prob, **parameters).step

    sample_key, warmup_key = jax.random.split(key)
    adapt = blackjax.window_adaptation(blackjax.nuts, log_prob)

    (last_state, parameters), _ = adapt.run(warmup_key, initial_state, num_samples)  # type: ignore # (blackjax problem)
    kernel = blackjax.nuts(log_prob, **parameters).step
    states, infos = inference_loop(sample_key, kernel, last_state, num_samples)
    return states.position


def test_nuts():
    def log_prob(d):
        x = d["xy"][0]
        y = d["xy"][1]
        z = d["z"]
        return -jnp.sum(x * x) - jnp.sum(y * y) - jnp.sum(z * z)

    d0 = {"xy": [jnp.zeros(2), jnp.zeros(5)], "z": jnp.ones(3)}

    key = jax.random.PRNGKey(0)
    samps = run_nuts(log_prob, key, d0, 1000)

    assert samps["xy"][0].shape == (1000, 2)
    assert samps["xy"][1].shape == (1000, 5)
    assert samps["z"].shape == (1000, 3)
