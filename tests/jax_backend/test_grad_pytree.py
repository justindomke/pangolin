"""Compare tree gradients and their compositions against JAX, eagerly and under JIT."""

import jax
import jax.numpy as jnp
import numpy as np

from pangolin import interface as pi, ir, jax_backend


def check(actual, expected):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for a, e in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
        assert a.shape == np.shape(e)
        np.testing.assert_allclose(a, e, rtol=2e-5, atol=2e-6)


def test_nested_arrays_with_unused_leaves_and_reused_subexpressions():
    def model(p):
        weights, bias = p["active"]
        scores = weights @ features + bias
        return pi.sum(scores * scores + pi.sin(scores), 0)

    def reference(p):
        weights, bias = p["active"]
        scores = weights @ feature_values + bias
        return jnp.sum(scores * scores + jnp.sin(scores))

    feature_values = jnp.array([0.2, -0.7, 1.1])
    features = pi.constant(feature_values)
    values = {"active": (jnp.arange(6.0).reshape(2, 3) / 5, jnp.array(0.3)),
              "unused": [jnp.array([2.0, 4.0]), ()]}
    arguments = jax.tree_util.tree_map(pi.constant, values)
    gradient = pi.grad(model)(arguments)

    def evaluate(p):
        bindings = dict(zip(jax.tree_util.tree_leaves(arguments), jax.tree_util.tree_leaves(p), strict=True))
        return jax_backend.ancestor_sample_graph(gradient, None, bindings)

    compiled = jax.jit(evaluate)
    for factor in [1.0, -0.5]:
        p = jax.tree_util.tree_map(lambda x: factor * x, values)
        expected = jax.grad(reference)(p)
        check(evaluate(p), expected)
        check(compiled(p), expected)


def test_aliases_captures_other_arguments_and_existing_dependencies():
    x = pi.constant(3.0)
    dependent = x * x
    results = (
        pi.grad(lambda p: p[0] * p[1])([x, x]),
        pi.grad(lambda p: p[0] * p[1])([x, dependent]),
        pi.grad(lambda p: p[0] * x + p[1] * dependent)((x, dependent)),
        pi.grad(lambda p, other: p[0] * other + p[1])([x, x], x),
        pi.grad(lambda p: p[0] + p[0] + p[1] * p[1])([x, x]),
    )

    def evaluate(value):
        return jax_backend.ancestor_sample_graph(results, None, {x: value})

    for value in [2.0, -3.0]:
        expected = ([value, value], [value**2, value], (value, value**2),
                    [value, 1.0], [2.0, 2 * value])
        check(evaluate(jnp.array(value)), expected)
        check(jax.jit(evaluate)(jnp.array(value)), expected)


def test_mixed_second_derivatives_of_tree_gradient():
    def model(p: dict[str, ir.RV]) -> ir.RV:
        return pi.exp(p["x"] * p["y"]) + p["x"] ** 3

    def reference(p):
        return jnp.exp(p["x"] * p["y"]) + p["x"] ** 3

    derivative = pi.grad(model)

    def dx(p: dict[str, ir.RV]) -> ir.RV:
        return derivative(p)["x"]

    def dy(p: dict[str, ir.RV]) -> ir.RV:
        return derivative(p)["y"]

    arguments = {"x": pi.constant(0.2), "y": pi.constant(0.3)}
    results = {"x": pi.grad(dx)(arguments), "y": pi.grad(dy)(arguments)}

    def evaluate(p):
        return jax_backend.ancestor_sample_graph(results, None, {arguments[k]: v for k, v in p.items()})

    for x, y in [(0.2, 0.3), (-0.7, 0.8)]:
        values = {"x": jnp.array(x), "y": jnp.array(y)}
        expected = jax.hessian(reference)(values)
        check(evaluate(values), expected)
        check(jax.jit(evaluate)(values), expected)


def test_tree_grad_through_scan_vmap_and_log_prob():
    def model(p: dict[str, ir.RV], data: ir.RV) -> ir.RV:
        path = pi.scan(lambda carry, x: p["rate"] * carry + x, 4)(p["start"], data)
        scores = pi.vmap(lambda x: pi.log_prob(pi.normal)(x, p["mean"], pi.exp(p["log_scale"])))(path)
        return pi.sum(scores, 0)

    def reference(p, data):
        def step(carry, x):
            next_value = p["rate"] * carry + x
            return next_value, next_value

        _, path = jax.lax.scan(step, p["start"], data)
        return jnp.sum(jax.scipy.stats.norm.logpdf(path, p["mean"], jnp.exp(p["log_scale"])))

    p = {k: pi.constant(0.0) for k in ["rate", "start", "mean", "log_scale"]}
    data = pi.normal([0.0] * 4, 1)
    results = pi.grad(model)(p, data)

    def evaluate(values, observations):
        return jax_backend.ancestor_sample_graph(results, None, {**{p[k]: v for k, v in values.items()}, data: observations})

    compiled = jax.jit(evaluate)
    for rate in [0.2, -0.8]:
        values = {"rate": jnp.array(rate), "start": jnp.array(0.4),
                  "mean": jnp.array(-0.2), "log_scale": jnp.array(0.3)}
        observations = jnp.array([-1.0, 0.3, 0.6, -0.2])
        expected = jax.grad(reference)(values, observations)
        check(evaluate(values, observations), expected)
        check(compiled(values, observations), expected)


def test_vmap_of_tree_grad_matches_grad_of_summed_vmapped_function():
    def score(p: dict[str, ir.RV], x: ir.RV) -> ir.RV:
        return pi.cdf(pi.normal)(x, p["mean"], pi.exp(p["log_scale"]))

    def total(p: dict[str, ir.RV], x: ir.RV) -> ir.RV:
        return pi.sum(pi.vmap(score, (None, 0))(p, x), 0)

    p = {"mean": pi.constant(0.3), "log_scale": pi.constant(-0.2)}
    x = pi.constant([-0.4, 0.5, 1.2])
    mapped = pi.vmap(pi.grad(score), (None, 0))(p, x)
    results = (mapped, pi.grad(total)(p, x))
    values = {"mean": jnp.array(0.3), "log_scale": jnp.array(-0.2)}
    observations = jnp.array([-0.4, 0.5, 1.2])

    def reference(p, x):
        return jax.scipy.special.ndtr((x - p["mean"]) / jnp.exp(p["log_scale"]))

    expected_mapped = jax.vmap(jax.grad(reference), (None, 0))(values, observations)
    expected = (expected_mapped, jax.grad(lambda p: jnp.sum(jax.vmap(reference, (None, 0))(p, observations)))(values))

    def evaluate():
        return jax_backend.ancestor_sample_graph(results, None)

    check(evaluate(), expected)
    check(jax.jit(evaluate)(), expected)
