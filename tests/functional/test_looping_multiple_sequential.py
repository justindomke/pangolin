"""Nested recurrences with separately constructed random initialization arrays."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import distributions as dist

from pangolin import dag, interface as pi, ir, jax_backend
from pangolin.functional import loop


def evaluate(output):
    return jax.jit(jax_backend.ancestor_sampler(output))(jax.random.key(7))


def test_two_random_initializers_joint_density_and_gradients():
    shared = pi.normal(0.0, 1.0)
    calls = []

    def init_t(s):
        calls.append("t")
        return pi.normal(shared + s, 0.4)

    def init_s(t):
        calls.append("s")
        latent = pi.normal(shared + t, 0.5)
        return pi.normal(latent, 0.6)

    def model(prev, t, s):
        if s is None:
            return init_s(t)
        if t is None:
            return init_t(s)
        calls.append("body")
        return pi.normal(0.2 * prev[t - 1, s] + 0.3 * prev[t, s - 1] + shared, 0.7)

    output = loop(model, interface=pi.default, s=[3], t=[2])
    assert calls == ["t", "s", "body"]
    nodes = [rv for rv in dag.upstream_nodes(output) if rv.op.random]
    assert [rv.shape for rv in nodes] == [(), (3,), (2,), (2,), (2, 3)]
    assert nodes[0] is shared and nodes[-1] is output
    values = (
        jnp.array(0.1),
        jnp.array([0.2, 0.8, 1.9]),
        jnp.array([-0.1, 1.2]),
        jnp.array([0.2, 0.9]),
        jnp.array([[0.3, -0.1, 0.8], [0.4, 0.2, 0.6]]),
    )

    def reference(v):
        z, initial_t, latent_s, initial_s, x = v
        old_t = jnp.concatenate((initial_t[None, :], x[:-1]), axis=0)
        old_s = jnp.concatenate((initial_s[:, None], x[:, :-1]), axis=1)
        return (
            dist.Normal(0, 1).log_prob(z)
            + dist.Normal(z + jnp.arange(3), 0.4).log_prob(initial_t).sum()
            + dist.Normal(z + jnp.arange(2), 0.5).log_prob(latent_s).sum()
            + dist.Normal(latent_s, 0.6).log_prob(initial_s).sum()
            + dist.Normal(0.2 * old_t + 0.3 * old_s + z, 0.7).log_prob(x).sum()
        )

    density = jax_backend.ancestor_log_prob(tuple(nodes))
    np.testing.assert_allclose(density(values), reference(values), rtol=1e-6)
    np.testing.assert_allclose(jax.jit(density)(values), reference(values), rtol=1e-6)
    actual = jax.jit(jax.grad(density))(values)
    expected = jax.grad(reference)(values)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_vectorized_axes_before_between_and_after_scans():
    def init_t(i, j, s, k):
        return i + 2.0 * j + 3 * s + k

    def init_s(i, t, j, k):
        return 10.0 * i + t + j + k

    def model(prev, i, t, j, s, k):
        if t is None:
            return init_t(i, j, s, k)
        if s is None:
            return init_s(i, t, j, k)
        old_t = pi.sum(prev[i, t - 1, :, s, 1 - k], 0)
        old_s = prev[i, t, j, s - 1, k]
        return 0.2 * old_t + 0.4 * old_s + 1.0

    output = loop(model, interface=pi.default, i=2, t=[2], j=3, s=[3], k=2)
    expected = np.zeros((2, 2, 3, 3, 2))
    for i in range(2):
        for t in range(2):
            for j in range(3):
                for s in range(3):
                    for k in range(2):
                        old_t = (
                            i + 2 * np.arange(3) + 3 * s + 1 - k
                            if t == 0
                            else expected[i, t - 1, :, s, 1 - k]
                        )
                        old_s = 10 * i + t + j + k if s == 0 else expected[i, t, j, s - 1, k]
                        expected[i, t, j, s, k] = 0.2 * old_t.sum() + 0.4 * old_s + 1
    np.testing.assert_allclose(evaluate(output), expected, rtol=1e-6)


def test_three_sequential_axes_share_a_captured_initializer():
    shared = pi.normal(0.0, 1.0)

    def model(prev, t, s, u):
        if t is None:
            return shared
        if s is None:
            return shared
        if u is None:
            return shared
        return 0.2 * prev[t - 1, s, u] + 0.3 * prev[t, s - 1, u] + 0.4 * prev[t, s, u - 1] + 1

    output = loop(
        model,
        interface=pi.default,
        t=[2],
        s=[3],
        u=[2],
    )
    assert [rv for rv in dag.upstream_nodes(output) if rv.op.random] == [shared]
    z, actual = evaluate((shared, output))
    expected = np.zeros((2, 3, 2))
    for t, s, u in np.ndindex(expected.shape):
        a = z if t == 0 else expected[t - 1, s, u]
        b = z if s == 0 else expected[t, s - 1, u]
        c = z if u == 0 else expected[t, s, u - 1]
        expected[t, s, u] = 0.2 * a + 0.3 * b + 0.4 * c + 1
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_matrix_output_and_singleton_axes():
    start = pi.constant([[1.0, 2.0], [3.0, 4.0]])

    def model(prev, t, s):
        if t is None:
            return start
        if s is None:
            return start
        return prev[t - 1, s, :, :] + prev[t, s - 1, :, :]

    for T, S in ((1, 1), (1, 3), (2, 1)):
        output = loop(model, interface=pi.default, t=[T], s=[S])
        expected = np.zeros((T, S, 2, 2))
        for t, s in np.ndindex(T, S):
            old_t = np.array([[1, 2], [3, 4]]) if t == 0 else expected[t - 1, s]
            old_s = np.array([[1, 2], [3, 4]]) if s == 0 else expected[t, s - 1]
            expected[t, s] = old_t + old_s
        assert output.shape == expected.shape
        np.testing.assert_allclose(evaluate(output), expected)


def test_log_prob_gradient_through_both_initializers():
    values = jnp.array([[0.3, 0.8, -0.2], [0.5, 0.2, 0.9]])

    def distribution(weight):
        def model(prev, t, s):
            if t is None:
                return weight * (s + 1)
            if s is None:
                return weight * weight + t
            return pi.normal(weight * prev[t - 1, s] + 0.4 * prev[t, s - 1], 0.7)

        return loop(
            model,
            interface=pi.default,
            t=[2],
            s=[3],
        )

    density = pi.log_prob(distribution)
    value_rv, weight = pi.constant(values), pi.constant(0.3)
    actual = evaluate((density(value_rv, weight), pi.grad(density, 1)(value_rv, weight)))

    def reference(w):
        old_t = jnp.concatenate(((w * (jnp.arange(3) + 1))[None, :], values[:-1]), axis=0)
        old_s = jnp.concatenate(((w * w + jnp.arange(2))[:, None], values[:, :-1]), axis=1)
        return dist.Normal(w * old_t + 0.4 * old_s, 0.7).log_prob(values).sum()

    np.testing.assert_allclose(actual, (reference(0.3), jax.grad(reference)(0.3)), rtol=1e-5)


def test_each_sequential_carry_must_be_used():
    for body in (
        lambda prev, t, s: prev[t - 1, s] + 1,
        lambda prev, t, s: prev[t, s - 1] + 1,
    ):
        with pytest.raises(ValueError, match="depend on its carry"):
            loop(
                body,
                interface=pi.default,
                t=[2, pi.constant(0)],
                s=[3, pi.constant(0)],
            )


def test_each_initializer_must_match_shape_and_interface():
    def body(prev, t, s):
        if t is None:
            return pi.constant(0.0)
        if s is None:
            return other.constant(0.0)
        return prev[t - 1, s] + prev[t, s - 1]

    with pytest.raises(ValueError, match="match carry shape"):

        def model(prev, t, s):
            if t is None:
                return pi.constant([0.0, 1.0])
            if s is None:
                return pi.constant(0.0)
            return prev[t - 1, s, :] + prev[t, s - 1]

        loop(
            model,
            interface=pi.default,
            t=[2],
            s=[3],
        )

    other = pi.Interface()
    with pytest.raises(ValueError, match="interface must match"):
        loop(
            body,
            interface=pi.default,
            t=[2],
            s=[3],
        )


def test_later_invalid_axis_is_rejected_before_any_initializer_runs():
    def initial(s):
        pytest.fail("no initializer should run")

    with pytest.raises(ValueError, match="must be positive"):

        def model(prev, t, s):
            if t is None:
                return initial(s)
            if s is None:
                return initial(t)
            return prev[t - 1, s] + prev[t, s - 1]

        loop(
            model,
            interface=pi.default,
            t=[2],
            s=[0],
        )


def test_random_initializers_and_nested_scan_innovations_are_independent():
    def model(prev, t, s):
        if t is None:
            return pi.normal(0.0, 1.0)
        if s is None:
            return pi.normal(0.0, 1.0)
        return pi.normal(0.2 * prev[t - 1, s] + 0.3 * prev[t, s - 1], 1.0)

    output = loop(
        model,
        interface=pi.default,
        t=[2],
        s=[3],
    )
    nodes = tuple(rv for rv in dag.upstream_nodes(output) if rv.op.random)
    assert [rv.shape for rv in nodes] == [(3,), (2,), (2, 3)]
    sampler = jax_backend.ancestor_sampler(nodes)
    a, b, x = jax.jit(jax.vmap(sampler))(jax.random.split(jax.random.key(9), 4000))
    old_t = jnp.concatenate((a[:, None, :], x[:, :-1, :]), axis=1)
    old_s = jnp.concatenate((b[:, :, None], x[:, :, :-1]), axis=2)
    innovations = x - 0.2 * old_t - 0.3 * old_s
    independent = np.concatenate((a, b, innovations.reshape(4000, 6)), axis=1)
    np.testing.assert_allclose(independent.mean(axis=0), 0.0, atol=0.06)
    np.testing.assert_allclose(np.cov(independent, rowvar=False), np.eye(11), atol=0.08)
