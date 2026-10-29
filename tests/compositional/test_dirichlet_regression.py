"""Compare batched Dirichlet densities with softmax-valued observations and parameter curvature against JAX."""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from pangolin import interface as pi, ir, jax_backend

from .helpers import assert_matches


def pangolin_likelihood(raw: ir.RV, logits: ir.RV) -> ir.RV:
    values = pi.vmap(pi.softmax)(logits)

    def observation(row: ir.RV) -> ir.RV:
        return pi.dirichlet(pi.exp(raw + 0.2 * row) + 1.0)

    return pi.log_prob(pi.vmap(observation))(values, logits)


def jax_likelihood(raw: jax.Array, logits: jax.Array) -> jax.Array:
    alpha = jnp.exp(raw + 0.2 * logits) + 1.0
    log_values = jax.nn.log_softmax(logits)
    normalization = gammaln(jnp.sum(alpha, axis=1)) - jnp.sum(gammaln(alpha), axis=1)
    return jnp.sum(normalization + jnp.sum((alpha - 1) * log_values, axis=1))


def test_dirichlet_regression():
    r = jnp.array([-0.4, 0.2, 0.6])
    x = jnp.array([[0.2, -0.7, 0.5], [-0.3, 0.4, 0.1]])
    direction = jnp.array([0.3, -0.6, 0.9])
    raw, logits = pi.constant(r), pi.constant(x)
    score = pi.grad(pangolin_likelihood)
    results = (
        pangolin_likelihood(raw, logits),
        score(raw, logits),
        pi.grad(pangolin_likelihood, 1)(raw, logits),
        pi.vjp(score)(pi.constant(direction), raw, logits),
    )

    def evaluate(parameter: jax.Array):
        return jax_backend.ancestor_sample_graph(results, None, {raw: parameter})

    compiled = jax.jit(evaluate)
    for parameter in (r, r + 0.3):
        _, pullback = jax.vjp(lambda a: jax.grad(jax_likelihood)(a, x), parameter)
        expected = (
            jax_likelihood(parameter, x),
            jax.grad(jax_likelihood)(parameter, x),
            jax.grad(jax_likelihood, 1)(parameter, x),
            pullback(direction)[0],
        )
        assert_matches(evaluate(parameter), expected, rtol=5e-5, atol=5e-6)
        assert_matches(compiled(parameter), expected, rtol=5e-5, atol=5e-6)
