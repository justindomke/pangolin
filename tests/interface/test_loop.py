"""Interface loop coercion, configuration, sharing, and functional equivalence."""

import jax
import numpy as np
import pytest

from pangolin import dag, functional, interface as pi, ir, jax_backend


def evaluate(x):
    return jax.jit(jax_backend.ancestor_sampler(x))(jax.random.key(0))


def test_default_export_is_bound_to_default_interface():
    assert pi.loop.__self__ is pi.default
    assert "loop" in pi.__all__


@pytest.mark.parametrize("decorator", [False, True])
def test_vectorized_coordinates_and_numeric_output_use_configured_interface(decorator):
    configured = pi.Interface()
    seen = []

    def body(i, j):
        seen.extend((i, j))
        return [1.0, 2.0]

    if decorator:

        @configured.loop
        def x(i=2, j=3):
            return body(i, j)

    else:
        x = configured.loop(body, j=3, i=2)
    assert x.shape == (2, 3, 2)
    assert x.interface is configured
    assert all(v.shape == () and v.interface is configured for v in seen)
    np.testing.assert_array_equal(evaluate(x), np.broadcast_to([1.0, 2.0], (2, 3, 2)))


def test_default_wrapper_matches_functional_loop():
    def body(prev, i, t):
        return pi.normal(prev[i, t - 1] + i, 1.0)

    x = pi.loop(body, i=2, t=[3, 0.0])
    y = functional.loop(body, interface=pi.default, i=2, t=[3, pi.constant(0.0)])
    assert x.op == y.op
    assert all(ir.rv_equal(a, b) for a, b in zip(x.parents, y.parents, strict=True))


@pytest.mark.parametrize("initializer", [0.0, np.float32(0.0), np.array(0.0)])
def test_value_initializers_are_coerced(initializer):
    configured = pi.Interface()

    def body(prev, i, t):
        assert i.interface is t.interface is configured
        assert prev[i, t - 1].interface is configured
        return prev[i, t - 1] + i + 1.0

    x = configured.loop(body, i=2, t=[3, initializer])
    assert x.interface is configured
    np.testing.assert_array_equal(evaluate(x), [[1, 2, 3], [2, 4, 6]])


def test_array_initializer_is_per_coordinate_state():
    x = pi.loop(lambda prev, i, t: prev[i, t - 1, :] + i + 1.0, i=2, t=[3, [1.0, 2.0]])
    assert x.shape == (2, 3, 2)
    expected = (
        np.array([1.0, 2.0]) + np.arange(1, 3)[:, None, None] * np.arange(1, 4)[None, :, None]
    )
    np.testing.assert_array_equal(evaluate(x), expected)


def test_none_initializers_coerce_results_for_both_axes():
    configured = pi.Interface()

    @configured.loop
    def x(prev, t=[2], s=[3, None]):
        if t is None:
            assert s.interface is configured
            return 1.0
        if s is None:
            assert t.interface is configured
            return 2.0
        return prev[t - 1, s] + prev[t, s - 1]

    assert x.interface is configured
    np.testing.assert_array_equal(evaluate(x), [[3, 4, 5], [5, 9, 14]])


def test_captured_random_initializer_preserved_by_identity():
    start = pi.normal(0.0, 1.0)
    x = pi.loop(lambda prev, i, t: prev[i, t - 1] + 1.0, i=2, t=[3, start])
    assert [v for v in dag.upstream_nodes(x) if v.op.random] == [start]
    initial, values = evaluate((start, x))
    np.testing.assert_allclose(
        values, np.broadcast_to(initial + np.arange(1, 4), (2, 3)), rtol=1e-6
    )


def test_existing_rv_output_and_initializers_keep_their_interface():
    configured, other = pi.Interface(), pi.Interface()
    start = other.constant(0.0)
    for initializer in (start, None):

        def body(prev, i, t):
            if t is None:
                return start
            return other.add(prev[i, t - 1], i + 1.0)

        x = configured.loop(body, i=2, t=[3, initializer])
        assert x.interface is other
        assert start in dag.upstream_nodes(x)
        np.testing.assert_array_equal(evaluate(x), [[1, 2, 3], [2, 4, 6]])
    plain = ir.RV(ir.Constant(1.0))
    assert configured.loop(lambda: plain) is plain
    x = configured.loop(lambda i: ir.RV(ir.Add(), i, plain), i=2)
    assert x.interface is ir.NO_INTERFACE


def test_no_axes_coerces_the_output():
    x = pi.loop(lambda: np.array([1, 2]))
    assert x.op == ir.Constant([1, 2])
    assert x.interface is pi.default


def test_declaration_uses_current_configuration():
    configured = pi.Interface(broadcasting="off")
    with configured.overload(broadcasting="simple"):

        @configured.loop
        def x(i=2):
            return i + [1, 2, 3]

    assert x.shape == (2, 3)
    assert x.interface is configured
    assert configured._effective_config.broadcasting == "off"


def test_nested_interface_loops_inside_initializer():
    @pi.loop
    def x(prev, t=[3]):
        if t is None:
            return pi.loop(lambda i: i + 1.0, i=2)
        return prev[t - 1, :] + 1.0

    np.testing.assert_array_equal(evaluate(x), [[2, 3], [3, 4], [4, 5]])


def test_coercion_wrappers_preserve_signature_validation():
    with pytest.raises(ValueError, match="missing size for axis 'j'"):
        pi.loop(lambda i, j: i + j, i=2)
    with pytest.raises(ValueError, match="unknown axis 'k'"):
        pi.loop(lambda i: i + 1, i=2, k=3)
    with pytest.raises(TypeError, match="cannot mix parameter defaults"):
        pi.loop(lambda i=2: i + 1, i=2)
    with pytest.raises(TypeError, match="initializer must be an RV, a numeric value, or None"):
        pi.loop(lambda prev, i, t: prev[i, t - 1] + 1.0, i=2, t=[3, lambda: 0.0])
    with pytest.raises(TypeError, match="expected a callable"):
        pi.loop(1, i=2)


def test_prev_reads_and_unused_carries_remain_rejected():
    with pytest.raises(ValueError, match="prev cannot be read during initialization"):
        pi.loop(lambda prev, t: prev[0] if t is None else prev[t - 1] + 1, t=[3])
    with pytest.raises(ValueError, match="depend on its carry"):
        pi.loop(lambda prev, t: 1.0, t=[3, 0.0])


def test_coercion_does_not_overwrite_foreign_interface():
    other = pi.Interface()
    with pytest.raises(ValueError, match="interface must match"):
        pi.loop(lambda prev, t: pi.add(prev[t - 1], 1.0), t=[3, other.constant(0.0)])


def test_existing_vectorized_output_remains_rejected():
    start = pi.normal(0.0, 1.0)
    with pytest.raises(ValueError, match="every output must be created"):
        pi.loop(lambda i: start, i=2)


@pytest.mark.parametrize("decorator", [False, True])
@pytest.mark.parametrize("container", [list, tuple])
def test_callable_initializer_rejected_before_body_or_initializer_runs(decorator, container):
    def initial():
        pytest.fail("initializer must not run")

    def body(prev, t):
        pytest.fail("body must not run")

    spec = container((3, initial))
    with pytest.raises(TypeError, match="initializer must be an RV, a numeric value, or None"):
        if decorator:

            @pi.loop
            def x(prev, t=spec):
                return body(prev, t)

        else:
            pi.loop(body, t=spec)
