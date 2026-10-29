"""Compare batched random Scans with carry-dependent noise scales against explicit conditional densities."""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from pangolin import interface as pi, ir, jax_backend

from .helpers import assert_matches


def pangolin_likelihood(log_scale: ir.RV, values: ir.RV, starts: ir.RV) -> ir.RV:
    def trajectory(start: ir.RV) -> ir.RV:
        def step(previous: ir.RV) -> ir.RV:
            return pi.normal(0.6 * previous, pi.exp(log_scale + 0.15 * previous))

        return pi.scan(step, 4)(start)

    return pi.log_prob(pi.vmap(trajectory))(values, starts)


def jax_likelihood(log_scale: jax.Array, values: jax.Array, starts: jax.Array) -> jax.Array:
    previous = jnp.concatenate((starts[:, None], values[:, :-1]), axis=1)
    return jnp.sum(norm.logpdf(values, 0.6 * previous, jnp.exp(log_scale + 0.15 * previous)))


def test_recurrent_volatility():
    observations = jnp.array([[0.2, -0.4, 0.1, 0.8], [-0.6, 0.3, 0.5, -0.2]])
    initial = jnp.array([-0.3, 0.7])
    log_scale, values, starts = pi.constant(-0.2), pi.constant(observations), pi.constant(initial)
    results = (
        pangolin_likelihood(log_scale, values, starts),
        pi.grad(pangolin_likelihood)(log_scale, values, starts),
        pi.grad(pangolin_likelihood, 1)(log_scale, values, starts),
        pi.grad(pangolin_likelihood, 2)(log_scale, values, starts),
        pi.grad(pi.grad(pangolin_likelihood))(log_scale, values, starts),
    )

    def evaluate(scale: jax.Array):
        return jax_backend.ancestor_sample_graph(results, None, {log_scale: scale})

    compiled = jax.jit(evaluate)
    for scale in (jnp.asarray(-0.5), jnp.asarray(0.2)):
        args = (scale, observations, initial)
        expected = (
            jax_likelihood(*args),
            jax.grad(jax_likelihood)(*args),
            jax.grad(jax_likelihood, 1)(*args),
            jax.grad(jax_likelihood, 2)(*args),
            jax.grad(jax.grad(jax_likelihood))(*args),
        )
        assert_matches(evaluate(scale), expected, rtol=4e-5)
        assert_matches(compiled(scale), expected, rtol=4e-5)
