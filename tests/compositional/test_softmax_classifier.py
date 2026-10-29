"""Compare a vectorized softmax classifier's matrix gradient and curvature with JAX."""

import jax
import jax.numpy as jnp

from pangolin import interface as pi, ir, jax_backend

from .helpers import assert_matches


def pangolin_likelihood(weights: ir.RV, design: ir.RV, labels: ir.RV) -> ir.RV:
    def classify(row: ir.RV) -> ir.RV:
        return pi.categorical(pi.softmax(weights @ row))

    return pi.log_prob(pi.vmap(pi.composite(classify, simplify=False)))(labels, design)


def jax_likelihood(weights: jax.Array, design: jax.Array, labels: jax.Array) -> jax.Array:
    logits = design @ weights.T
    return jnp.sum(jax.nn.log_softmax(logits)[jnp.arange(len(labels)), labels])


def test_softmax_classifier():
    x = jnp.array([[1.0, -0.8], [1.0, 0.3], [1.0, 1.1], [1.0, -0.2]])
    y = jnp.array([2, 0, 1, 2])
    w = jnp.array([[0.2, -0.5], [-0.3, 0.7], [0.4, 0.1]])
    direction = jnp.array([[0.1, -0.4], [0.8, 0.2], [-0.3, 0.5]])
    weights, design, labels = pi.constant(w), pi.constant(x), pi.constant(y)
    score = pi.grad(pangolin_likelihood)
    results = (
        pangolin_likelihood(weights, design, labels),
        score(weights, design, labels),
        pi.vjp(score)(pi.constant(direction), weights, design, labels),
    )

    def evaluate(matrix: jax.Array):
        return jax_backend.ancestor_sample_graph(results, None, {weights: matrix})

    compiled = jax.jit(evaluate)
    for matrix in (w, w + jnp.array([[0.3], [-0.1], [0.2]])):
        _, pullback = jax.vjp(lambda m: jax.grad(jax_likelihood)(m, x, y), matrix)
        expected = (
            jax_likelihood(matrix, x, y),
            jax.grad(jax_likelihood)(matrix, x, y),
            pullback(direction)[0],
        )
        assert_matches(evaluate(matrix), expected)
        assert_matches(compiled(matrix), expected)
