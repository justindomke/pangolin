"""Axis order, initialization ownership, indexing, and rejection rules for nested recurrences."""

import jax
import numpy as np
import pytest

from pangolin import dag, interface as pi, ir, jax_backend
from pangolin.functional import loop


def evaluate(output):
    return jax.jit(jax_backend.ancestor_sampler(output))(jax.random.key(31))


def test_all_six_orders_of_two_sequential_axes_and_one_vectorized_axis():
    def its(prev, i, t, s):
        if t is None:
            return 10.0 * i + s
        if s is None:
            return 20.0 * i + t
        return 0.2 * prev[i, t - 1, s] + 0.3 * prev[i, t, s - 1] + i + 1

    def ist(prev, i, s, t):
        if t is None:
            return 10.0 * i + s
        if s is None:
            return 20.0 * i + t
        return 0.2 * prev[i, s, t - 1] + 0.3 * prev[i, s - 1, t] + i + 1

    def tis(prev, t, i, s):
        if t is None:
            return 10.0 * i + s
        if s is None:
            return 20.0 * i + t
        return 0.2 * prev[t - 1, i, s] + 0.3 * prev[t, i, s - 1] + i + 1

    def tsi(prev, t, s, i):
        if t is None:
            return 10.0 * i + s
        if s is None:
            return 20.0 * i + t
        return 0.2 * prev[t - 1, s, i] + 0.3 * prev[t, s - 1, i] + i + 1

    def sit(prev, s, i, t):
        if t is None:
            return 10.0 * i + s
        if s is None:
            return 20.0 * i + t
        return 0.2 * prev[s, i, t - 1] + 0.3 * prev[s - 1, i, t] + i + 1

    def sti(prev, s, t, i):
        if t is None:
            return 10.0 * i + s
        if s is None:
            return 20.0 * i + t
        return 0.2 * prev[s, t - 1, i] + 0.3 * prev[s - 1, t, i] + i + 1

    expected = np.zeros((2, 3, 4))
    for i, t, s in np.ndindex(expected.shape):
        a = 10 * i + s if t == 0 else expected[i, t - 1, s]
        b = 20 * i + t if s == 0 else expected[i, t, s - 1]
        expected[i, t, s] = 0.2 * a + 0.3 * b + i + 1

    cases = (
        (its, (0, 1, 2)),
        (ist, (0, 2, 1)),
        (tis, (1, 0, 2)),
        (tsi, (1, 2, 0)),
        (sit, (2, 0, 1)),
        (sti, (2, 1, 0)),
    )
    for body, permutation in cases:
        output = loop(body, interface=pi.default, s=[4], i=2, t=[3])
        assert output.shape == tuple(expected.shape[k] for k in permutation)
        np.testing.assert_allclose(evaluate(output), expected.transpose(permutation), rtol=1e-6)


def test_suffix_predecessor_is_ordinary_negative_indexing_at_first_step():
    def model(prev, t, s):
        if t is None:
            return 10.0 + s
        if s is None:
            return 20.0 + t
        return 0.2 * prev[t - 1, s - 1] + 0.3 * prev[t, s - 1]

    output = loop(
        model,
        interface=pi.default,
        t=[2],
        s=[3],
    )
    expected = np.zeros((2, 3))
    for t, s in np.ndindex(expected.shape):
        old_t = np.arange(3) + 10 if t == 0 else expected[t - 1]
        old_s = 20 + t if s == 0 else expected[t, s - 1]
        expected[t, s] = 0.2 * old_t[s - 1] + 0.3 * old_s
    np.testing.assert_allclose(evaluate(output), expected, rtol=1e-6)


def test_complete_outer_carry_is_available_without_suffix_indices():
    def model(prev, t, s):
        if t is None:
            return s + 1.0
        if s is None:
            return t + 2.0
        return 0.2 * pi.sum(prev[t - 1], 0) + 0.3 * prev[t, s - 1]

    output = loop(
        model,
        interface=pi.default,
        t=[2],
        s=[3],
    )
    expected = np.zeros((2, 3))
    for t, s in np.ndindex(expected.shape):
        old_t = np.arange(3) + 1 if t == 0 else expected[t - 1]
        old_s = t + 2 if s == 0 else expected[t, s - 1]
        expected[t, s] = 0.2 * old_t.sum() + 0.3 * old_s
    np.testing.assert_allclose(evaluate(output), expected, rtol=1e-6)


def test_initializers_may_return_their_coordinate_directly():
    def model(prev, t, s):
        if t is None:
            return s
        if s is None:
            return t
        return prev[t - 1, s] + prev[t, s - 1]

    output = loop(
        model,
        interface=pi.default,
        t=[2],
        s=[3],
    )
    np.testing.assert_array_equal(evaluate(output), [[0, 1, 3], [1, 2, 5]])
    assert not any(rv.op.random for rv in dag.upstream_nodes(output))


def test_random_nested_loops_and_deterministic_reduction_inside_initializer():
    def init_t(s):
        def model(prev, u, v):
            if u is None:
                return pi.normal(s + v, 1.0)
            if v is None:
                return pi.normal(s - u, 1.0)
            return pi.normal(0.2 * prev[u - 1, v] + 0.3 * prev[u, v - 1] + s, 1.0)

        block = loop(
            model,
            interface=pi.default,
            u=[2],
            v=[2],
        )
        return pi.sum(pi.sum(block, 0), 0)

    def model_2(prev, t, s):
        if t is None:
            return init_t(s)
        if s is None:
            return pi.normal(t, 1.0)
        return 0.2 * prev[t - 1, s] + 0.3 * prev[t, s - 1]

    output = loop(
        model_2,
        interface=pi.default,
        t=[2],
        s=[3],
    )
    nodes = [rv for rv in dag.upstream_nodes(output) if rv.op.random]
    assert [rv.shape for rv in nodes] == [(3, 2), (3, 2), (3, 2, 2), (2,)]
    blocks, initial_s, actual = evaluate((nodes[2], nodes[3], output))
    initial_t = np.asarray(blocks).sum(axis=(1, 2))
    expected = np.zeros((2, 3))
    for t, s in np.ndindex(expected.shape):
        a = initial_t[s] if t == 0 else expected[t - 1, s]
        b = initial_s[t] if s == 0 else expected[t, s - 1]
        expected[t, s] = 0.2 * a + 0.3 * b
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_coordinate_interface_and_state_interface_are_independent():
    coordinate_interface = pi.Interface(broadcasting="off")
    state_interface = pi.Interface(broadcasting="numpy")
    seen = []

    def init_t(s):
        seen.append("t")
        assert s.interface is coordinate_interface
        return ir.RV(ir.Add(), s, pi.constant(1.0), interface=state_interface)

    def init_s(t):
        seen.append("s")
        assert t.interface is coordinate_interface
        return ir.RV(ir.Add(), t, pi.constant(2.0), interface=state_interface)

    def body(prev, t, s):
        if t is None:
            return init_t(s)
        if s is None:
            return init_s(t)
        seen.append("body")
        assert t.interface is s.interface is coordinate_interface
        a, b = prev[t - 1, s], prev[t, s - 1]
        assert a.interface is b.interface is state_interface
        return a + b

    output = loop(body, interface=coordinate_interface, t=[2], s=[3])
    assert seen == ["t", "s", "body"]
    assert output.interface is state_interface
    np.testing.assert_array_equal(evaluate(output), [[3, 5, 8], [6, 11, 19]])


def test_every_carry_in_three_axis_recurrence_must_contribute():
    bodies = (
        lambda prev, t, s, u: prev[t, s - 1, u] + prev[t, s, u - 1],
        lambda prev, t, s, u: prev[t - 1, s, u] + prev[t, s, u - 1],
        lambda prev, t, s, u: prev[t - 1, s, u] + prev[t, s - 1, u],
    )
    for body in bodies:
        with pytest.raises(ValueError, match="depend on its carry"):
            loop(
                body,
                interface=pi.default,
                t=[2, pi.constant(0.0)],
                s=[2, pi.constant(0.0)],
                u=[2, pi.constant(0.0)],
            )


def test_random_intermediates_in_inner_body_are_rejected():
    def body(prev, t, s):
        if t is None:
            return pi.constant(0.0)
        if s is None:
            return pi.constant(0.0)
        latent = pi.normal(prev[t - 1, s], 1.0)
        return pi.normal(latent + prev[t, s - 1], 1.0)

    with pytest.raises(ValueError, match="all but the last Op must be deterministic"):
        loop(
            body,
            interface=pi.default,
            t=[2],
            s=[3],
        )


def test_new_random_initializer_inside_recurrence_is_rejected():
    def body(prev, t, s):
        if t is None:
            return pi.constant(0.0)
        if s is None:
            return pi.constant(0.0)
        a, b = prev[t - 1, s], prev[t, s - 1]

        def model(prev, u):
            if u is None:
                return pi.normal(0.0, 1.0)
            return pi.normal(prev[u - 1] + a + b, 1.0)

        return loop(
            model,
            interface=pi.default,
            u=[2],
        )

    with pytest.raises(ValueError, match="all but the last Op must be deterministic"):
        loop(
            body,
            interface=pi.default,
            t=[2],
            s=[3],
        )


def test_invalid_predecessors_in_mixed_axes_are_rejected():
    readers = (
        lambda prev, t, j, s: prev[t + 0, j, s - 1],
        lambda prev, t, j, s: prev[t, j - 1, s],
        lambda prev, t, j, s: prev[t + 1, j, s],
        lambda prev, t, j, s: prev[t, j, s - 2],
    )
    for read in readers:
        with pytest.raises(ValueError, match="coordinate|sequential"):

            def model(prev, t, j, s):
                if t is None:
                    return pi.constant(0)
                if s is None:
                    return pi.constant(0)
                return read(prev, t, j, s)

            loop(
                model,
                interface=pi.default,
                t=[2],
                j=2,
                s=[3],
            )


def test_pytree_and_python_outputs_are_rejected_from_every_initializer():
    def valid(s):
        return pi.constant(0)

    for bad_output in (0.0, [pi.constant(0)], {"x": pi.constant(0)}):

        def invalid(s):
            return bad_output

        for init_t, init_s in ((valid, invalid), (invalid, valid)):

            def body(prev, t, s):
                if t is None:
                    return init_t(s)
                if s is None:
                    return init_s(t)
                return prev[t - 1, s] + prev[t, s - 1]

            with pytest.raises(TypeError, match="single RV"):
                loop(body, interface=pi.default, t=[2], s=[3])


def test_backend_rejects_integer_carries_that_become_floating_point():
    # IR shapes agree, but JAX requires each carry dtype to stay fixed.
    for t_start, s_start in ((0, 0.0), (0.0, 0)):

        def model(prev, t, s):
            if t is None:
                return pi.constant(t_start)
            if s is None:
                return pi.constant(s_start)
            return 0.2 * prev[t - 1, s] + 0.3 * prev[t, s - 1] + 1.0

        output = loop(
            model,
            interface=pi.default,
            t=[2],
            s=[3],
        )
        with pytest.raises(TypeError, match="carry input and carry output must have equal types"):
            evaluate(output)


def test_three_carries_allow_cross_reads_in_all_later_axes():
    def body(prev, t, s, u):
        if t is None:
            return 10.0 + 2 * s + u
        if s is None:
            return 20.0 + 3 * t + u
        if u is None:
            return 30.0 + 4 * t + s
        return (
            0.2 * prev[t - 1, 2 - s, 1 - u] + 0.3 * prev[t, s - 1, 1 - u] + 0.4 * prev[t, s, u - 1]
        )

    output = loop(
        body,
        interface=pi.default,
        t=[2],
        s=[3],
        u=[2],
    )
    expected = np.zeros((2, 3, 2))
    for t, s, u in np.ndindex(expected.shape):
        a = 10 + 2 * (2 - s) + 1 - u if t == 0 else expected[t - 1, 2 - s, 1 - u]
        b = 20 + 3 * t + 1 - u if s == 0 else expected[t, s - 1, 1 - u]
        c = 30 + 4 * t + s if u == 0 else expected[t, s, u - 1]
        expected[t, s, u] = 0.2 * a + 0.3 * b + 0.4 * c
    np.testing.assert_allclose(evaluate(output), expected, rtol=1e-6)


def test_callback_errors_propagate_without_corrupting_later_traces():
    class CallbackError(Exception):
        pass

    def fail_after_building_graph(s):
        pi.normal(s, 1.0)
        raise CallbackError("initializer failed")

    def fail_in_body(prev, t, s):
        if t is None:
            return pi.constant(1.0)
        if s is None:
            return pi.constant(1.0)
        prev[t - 1, s] + prev[t, s - 1]
        raise CallbackError("body failed")

    with pytest.raises(CallbackError, match="initializer failed"):

        def model(prev, t, s):
            if t is None:
                return fail_after_building_graph(s)
            if s is None:
                return pi.constant(1.0)
            return prev[t - 1, s] + prev[t, s - 1]

        loop(
            model,
            interface=pi.default,
            t=[2],
            s=[3],
        )
    with pytest.raises(CallbackError, match="body failed"):
        loop(
            fail_in_body,
            interface=pi.default,
            t=[2],
            s=[3],
        )

    def model_2(prev, t, s):
        if t is None:
            return pi.constant(1.0)
        if s is None:
            return pi.constant(1.0)
        return prev[t - 1, s] + prev[t, s - 1]

    output = loop(
        model_2,
        interface=pi.default,
        t=[2],
        s=[3],
    )
    np.testing.assert_array_equal(evaluate(output), [[2, 3, 4], [3, 6, 10]])
    assert not any(rv.op.random for rv in dag.upstream_nodes(output))
