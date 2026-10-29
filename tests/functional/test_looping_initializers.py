"""Equivalent loop spellings, initialization boundaries, and shared randomness."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import distributions as dist

from pangolin import dag, interface as pi, ir, jax_backend
from pangolin.functional import loop


def evaluate(x):
    return jax.jit(jax_backend.ancestor_sampler(x))(jax.random.key(0))


def test_list_and_tuple_initializers_are_equivalent():
    start = pi.constant(3)
    for spec in ([2, start], (2, start)):
        x = loop(lambda prev, t: prev[t - 1] + 1, interface=pi.default, t=spec)
        np.testing.assert_array_equal(evaluate(x), [4, 5])


def test_rv_initializer_matches_none_branch_and_stays_shared():
    start = pi.normal(0.0, 1.0)

    def body(prev, i, t):
        if t is None:
            return start
        return prev[i, t - 1] + i + 1

    direct = loop(body, interface=pi.default, i=2, t=[3, start])
    internal = loop(body, interface=pi.default, i=2, t=[3])
    assert direct.op == internal.op
    for x in (direct, internal):
        assert [v for v in dag.upstream_nodes(x) if v.op.random] == [start]
        initial, path = evaluate((start, x))
        np.testing.assert_allclose(
            path, initial + np.arange(1, 4)[None, :] * np.arange(1, 3)[:, None]
        )


def test_array_initializer_is_per_coordinate_state():
    start = pi.constant([2.0, 5.0])
    x = loop(lambda prev, i, t: prev[i, t - 1, :] + i + 1, interface=pi.default, i=3, t=[2, start])
    assert x.shape == (3, 2, 2)
    expected = (
        np.array([2.0, 5.0]) + np.arange(1, 4)[:, None, None] * np.arange(1, 3)[None, :, None]
    )
    np.testing.assert_array_equal(evaluate(x), expected)


def test_one_shared_rv_initializes_both_sequential_axes():
    start = pi.normal(0.0, 1.0)
    x = loop(
        lambda prev, t, s: prev[t - 1, s] + prev[t, s - 1],
        interface=pi.default,
        t=[2, start],
        s=[3, start],
    )
    assert [v for v in dag.upstream_nodes(x) if v.op.random] == [start]
    initial, values = evaluate((start, x))
    np.testing.assert_allclose(values, initial * np.array([[2, 3, 4], [3, 6, 10]]), rtol=1e-6)


def test_internal_initialization_spellings_and_value_are_equivalent():
    def body(prev, t):
        if t is None:
            return pi.constant(2)
        return prev[t - 1] + t + 1

    expected = loop(body, interface=pi.default, t=[3, pi.constant(2)])
    for spec in ([3], (3,), [3, None], (3, None)):
        x = loop(body, interface=pi.default, t=spec)
        assert x.op == expected.op
        np.testing.assert_array_equal(evaluate(x), [3, 5, 8])


def test_multiple_none_branches_with_vectorized_axes():
    calls = []

    def body(prev, i, t, j, s):
        calls.append((t is None, s is None))
        if t is None:
            return i + j + s + 1
        if s is None:
            return i + j + t + 2
        return prev[i, t - 1, j, s] + prev[i, t, j, s - 1]

    x = loop(body, interface=pi.default, s=[3], j=2, t=[2], i=2)
    assert calls == [(True, False), (False, True), (False, False)]
    expected = np.zeros((2, 2, 2, 3), dtype=int)
    for i in range(2):
        for t in range(2):
            for j in range(2):
                for s in range(3):
                    a = expected[i, t - 1, j, s] if t else i + j + s + 1
                    b = expected[i, t, j, s - 1] if s else i + j + t + 2
                    expected[i, t, j, s] = a + b
    np.testing.assert_array_equal(evaluate(x), expected)


def test_explicit_initializers_suppress_only_their_own_none_branch():
    calls = []

    def body(prev, t, s):
        assert t is not None
        calls.append(s is None)
        if s is None:
            return t + 2
        return prev[t - 1, s] + prev[t, s - 1]

    x = loop(body, interface=pi.default, t=[2, pi.constant(1)], s=[3])
    assert calls == [True, False]
    np.testing.assert_array_equal(evaluate(x), [[3, 4, 5], [6, 10, 15]])


def test_prev_reads_forbidden_during_initialization():
    def body(prev, t, s):
        if t is None:
            return prev[None, s - 1]
        if s is None:
            return pi.constant(0)
        return prev[t - 1, s] + prev[t, s - 1]

    with pytest.raises(ValueError, match="prev cannot be read during initialization"):
        loop(body, interface=pi.default, t=[2], s=[3])


def test_internal_initializer_can_build_nested_random_loop():
    def body(prev, i, t):
        if t is None:
            return pi.sum(loop(lambda j: pi.normal(i + j, 1.0), interface=pi.default, j=3), 0)
        return prev[i, t - 1] + 1.0

    x = loop(body, interface=pi.default, i=2, t=[3])
    nodes = [v for v in dag.upstream_nodes(x) if v.op.random]
    assert len(nodes) == 1 and nodes[0].shape == (2, 3)
    draws, values = evaluate((nodes[0], x))
    np.testing.assert_allclose(values, draws.sum(axis=1)[:, None] + np.arange(1, 4), atol=1e-6)


def test_random_none_initialization_density_and_gradient():
    z = pi.normal(0.0, 1.0)

    def body(prev, t, s):
        if t is None:
            return pi.normal(z + s, 0.5)
        if s is None:
            return pi.normal(z - t, 0.7)
        return pi.normal(0.2 * prev[t - 1, s] + 0.3 * prev[t, s - 1] + z, 0.8)

    a = loop(body, interface=pi.default, t=[2], s=[3])
    values = (
        jnp.array(0.2),
        jnp.array([0.1, 0.3, 0.7]),
        jnp.array([0.4, -0.1]),
        jnp.arange(6.0).reshape(2, 3) / 5,
    )

    def reference(v):
        z, a, b, x = v
        old_t = jnp.concatenate((a[None, :], x[:-1]), axis=0)
        old_s = jnp.concatenate((b[:, None], x[:, :-1]), axis=1)
        return (
            dist.Normal(0, 1).log_prob(z)
            + dist.Normal(z + jnp.arange(3), 0.5).log_prob(a).sum()
            + dist.Normal(z - jnp.arange(2), 0.7).log_prob(b).sum()
            + dist.Normal(0.2 * old_t + 0.3 * old_s + z, 0.8).log_prob(x).sum()
        )

    nodes = [v for v in dag.upstream_nodes(a) if v.op.random]
    by_shape = {v.shape: v for v in nodes}
    assert len(nodes) == 4
    density = jax_backend.ancestor_log_prob(tuple(by_shape[s] for s in [(), (3,), (2,), (2, 3)]))
    np.testing.assert_allclose(jax.jit(density)(values), reference(values), rtol=1e-6)
    actual, expected = jax.jit(jax.grad(density))(values), jax.grad(reference)(values)
    for x, y in zip(actual, expected, strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6)


def test_decorators_vectorized_sequential_and_no_axes():
    @partial(loop, interface=pi.default)
    def x(i=2, j=3):
        return 10 * i + j

    np.testing.assert_array_equal(evaluate(x), [[0, 1, 2], [10, 11, 12]])

    @partial(loop, interface=pi.default)
    def y(prev, t=[3]):
        if t is None:
            return pi.constant(2)
        return prev[t - 1] + 1

    np.testing.assert_array_equal(evaluate(y), [3, 4, 5])

    @partial(loop, interface=pi.default)
    def z():
        return pi.constant(7)

    assert z.op == ir.Constant(7)


def test_function_can_be_applied_at_different_sizes():
    def body(i):
        return i + 1

    first = loop(body, interface=pi.default, i=2)
    second = loop(body, interface=pi.default, i=3)
    np.testing.assert_array_equal(evaluate(first), [1, 2])
    np.testing.assert_array_equal(evaluate(second), [1, 2, 3])


@pytest.mark.parametrize("value", [0.0, 1, [1, 2], np.array([1, 2])])
def test_functional_does_not_coerce_initializers_or_none_results(value):
    with pytest.raises(TypeError, match="initializer must be an RV"):
        loop(lambda prev, t: prev[t - 1] + 1, interface=pi.default, t=[2, value])

    def body(prev, t):
        if t is None:
            return value
        return prev[t - 1] + 1

    with pytest.raises(TypeError, match="must return a single RV"):
        loop(body, interface=pi.default, t=[2])


def test_bad_later_callback_rejected_before_none_branch_runs():
    def body(prev, t, s):
        pytest.fail("body must not run")

    with pytest.raises(TypeError, match="initializer must be an RV or None"):
        loop(body, interface=pi.default, t=[2], s=[3, lambda: pi.constant(0)])


def test_none_branch_shape_and_interface_must_match_recurrence():
    def shape(prev, t):
        if t is None:
            return pi.constant([0.0, 0.0])
        return pi.sum(prev[t - 1, :], 0)

    with pytest.raises(ValueError, match="match carry shape"):
        loop(shape, interface=pi.default, t=[2])
    other = pi.Interface()

    def interface(prev, t):
        if t is None:
            return other.constant(0.0)
        return pi.add(prev[t - 1], 1.0)

    with pytest.raises(ValueError, match="interface must match"):
        loop(interface, interface=pi.default, t=[2])
