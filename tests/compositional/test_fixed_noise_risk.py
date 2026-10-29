"""Compare pathwise derivatives of a smoothed lognormal risk with existing random noise held fixed."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.special import ndtr

from pangolin import functional, interface as pi, ir, jax_backend

from .helpers import assert_matches


def pangolin_vmap(theta: ir.RV, noise: ir.RV) -> ir.RV:
    def cost(z: ir.RV) -> ir.RV:
        sample = pi.exp(theta[0] + pi.exp(theta[1]) * z)
        return pi.cdf(pi.normal)(sample, 1.3, 0.5)

    return pi.sum(pi.vmap(cost)(noise), 0) / 4


def pangolin_vfor(theta: ir.RV, noise: ir.RV) -> ir.RV:
    def cost(i: ir.RV) -> ir.RV:
        sample = pi.exp(theta[0] + pi.exp(theta[1]) * noise[i])
        return pi.cdf(pi.normal)(sample, 1.3, 0.5)

    return pi.sum(pi.vfor(cost, i=4), 0) / 4


def pangolin_loop(theta: ir.RV, noise: ir.RV) -> ir.RV:
    def cost(i: ir.RV) -> ir.RV:
        sample = pi.exp(theta[0] + pi.exp(theta[1]) * noise[i])
        return pi.cdf(pi.normal)(sample, 1.3, 0.5)

    return pi.sum(functional.loop(cost, interface=pi.default, i=4), 0) / 4


def jax_risk(theta: jax.Array, noise: jax.Array) -> jax.Array:
    samples = jnp.exp(theta[0] + jnp.exp(theta[1]) * noise)
    return jnp.mean(ndtr((samples - 1.3) / 0.5))


@pytest.mark.parametrize("risk", [pangolin_vmap, pangolin_vfor, pangolin_loop])
def test_fixed_noise_risk(risk: Callable[[ir.RV, ir.RV], ir.RV]):
    theta = pi.constant([0.2, -0.4])
    noise = pi.normal(pi.constant([0.0] * 4), 1)
    direction = jnp.array([0.4, -0.7])
    score = pi.grad(risk)
    results = (
        risk(theta, noise),
        score(theta, noise),
        pi.grad(risk, 1)(theta, noise),
        pi.vjp(score)(pi.constant(direction), theta, noise),
    )

    def evaluate(parameter: jax.Array, draws: jax.Array):
        return jax_backend.ancestor_sample_graph(results, None, {theta: parameter, noise: draws})

    compiled = jax.jit(evaluate)
    cases = [([0.2, -0.4], [-1.1, -0.3, 0.5, 1.2]), ([-0.3, -0.1], [0.8, -0.7, 0.1, -0.2])]
    for parameter, draws in cases:
        t, z = jnp.array(parameter), jnp.array(draws)
        _, pullback = jax.vjp(lambda a: jax.grad(jax_risk)(a, z), t)
        expected = (
            jax_risk(t, z),
            jax.grad(jax_risk)(t, z),
            jax.grad(jax_risk, 1)(t, z),
            pullback(direction)[0],
        )
        assert_matches(evaluate(t, z), expected)
        assert_matches(compiled(t, z), expected)
