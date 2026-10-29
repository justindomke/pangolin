"""Named vectorized loops return one RV and preserve random dependencies."""

from collections import namedtuple

import jax
import numpy as np
import pytest
from numpyro import distributions as dist

from pangolin import dag, functional, interface as pi, ir, jax_backend
from pangolin.functional.looping import loop


def evaluate(output):
    return jax.jit(jax_backend.ancestor_sampler(output))(jax.random.key(0))


def test_parameter_order_and_array_outputs_match_numpy_and_vfor():
    data = pi.constant(np.arange(24).reshape(2, 3, 4))

    def model(i, j):
        return data[i, j, :] + 100 * i + 10 * j

    result = loop(model, interface=pi.default, j=np.int64(3), i=2)
    expected = np.arange(24).reshape(2, 3, 4) + 100 * np.arange(2)[:, None, None]
    expected = expected + 10 * np.arange(3)[None, :, None]
    assert result.shape == (2, 3, 4)
    np.testing.assert_array_equal(evaluate(result), expected)
    assert ir.rv_equal(result, functional.vfor(model, interface=pi.default, i=2, j=3))


def test_shared_ancestors_and_multiple_random_nodes():
    shared = pi.normal(0, 1)

    def model(i, j):
        z = pi.normal(shared + i + j, 0.5)
        y = pi.normal(z, 0.2)
        return z + (y - z)

    result = loop(model, interface=pi.default, i=2, j=3)
    z, residual = result.parents
    y, alias = residual.parents
    assert alias is z
    assert y.parents[0] is z
    random_nodes = [rv for rv in dag.upstream_nodes(result) if rv.op.random]
    assert len(random_nodes) == 3
    assert shared in random_nodes
    assert result.shape == (2, 3)
    result_values, y_values = evaluate((result, y))
    np.testing.assert_allclose(result_values, y_values, atol=1e-6)


def test_nested_loop_preserves_row_randomness_and_matches_manual_density():
    def row(i):
        offset = pi.normal(i, 1)
        return loop(lambda j: pi.normal(offset + j, 0.5), interface=pi.default, j=3)

    cells = loop(row, interface=pi.default, i=2)
    random_nodes = [rv for rv in dag.upstream_nodes(cells) if rv.op.random]
    assert len(random_nodes) == 2
    offsets = next(rv for rv in random_nodes if rv is not cells)
    assert offsets.shape == (2,)
    assert cells.shape == (2, 3)
    offset_values = np.array([0.2, 0.8])
    cell_values = np.array([[0.4, 1.1, 2.0], [1.2, 1.5, 2.9]])
    density = jax_backend.ancestor_log_prob((offsets, cells))
    expected = dist.Normal(np.arange(2), 1).log_prob(offset_values).sum()
    expected += dist.Normal(offset_values[:, None] + np.arange(3), 0.5).log_prob(cell_values).sum()
    np.testing.assert_allclose(jax.jit(density)((offset_values, cell_values)), expected, rtol=1e-6)


def test_unused_axes_create_independent_draws_but_preserve_shared_capture():
    shared = pi.normal(0, 1)
    x = loop(lambda i, j: pi.normal(shared, 1), interface=pi.default, i=2, j=3)
    assert x.shape == (2, 3)
    assert x.parents[0] is shared
    assert x.op == ir.VMap(ir.VMap(ir.Normal(), (None, None), 3), (None, None), 2)
    sample = jax_backend.ancestor_sampler((shared, x))
    shared_values, values = jax.jit(jax.vmap(sample))(jax.random.split(jax.random.key(1), 4000))
    residuals = np.asarray(values - shared_values[:, None, None]).reshape(4000, 6)
    np.testing.assert_allclose(np.cov(residuals, rowvar=False), np.eye(6), atol=0.08)


def test_coordinate_and_output_interfaces_are_preserved():
    coordinates_interface = pi.Interface(broadcasting="off")
    output_interface = pi.Interface(broadcasting="numpy")
    seen = []

    def model(i, j):
        seen.append((i, j))
        assert i.shape == j.shape == ()
        assert i.interface is j.interface is coordinates_interface
        return ir.RV(ir.Add(), i, j, interface=output_interface)

    result = loop(model, interface=coordinates_interface, i=2, j=3)
    assert len(seen) == 1
    assert result.interface is output_interface
    np.testing.assert_array_equal(evaluate(result), [[0, 1, 2], [1, 2, 3]])


def test_no_axes_calls_directly_and_allows_existing_rvs():
    existing = pi.normal(0, 1)
    assert loop(lambda: existing, interface=pi.default) is existing


def test_missing_and_unknown_sizes_are_rejected_before_calling_body():
    def model(i, j):
        pytest.fail("invalid sizes should be rejected before tracing")

    with pytest.raises(ValueError, match="missing size for axis 'j'"):
        loop(model, interface=pi.default, i=2)
    with pytest.raises(ValueError, match="unknown axis 'k'"):
        loop(model, interface=pi.default, i=2, j=3, k=4)


def test_invalid_sizes_are_rejected():
    for size in (True, False, 2.5, "2", None, range(2)):
        with pytest.raises(TypeError, match="must be an integer"):
            loop(lambda i: i + 1, interface=pi.default, i=size)
    for size in (0, -2):
        with pytest.raises(ValueError, match="must be positive"):
            loop(lambda i: i + 1, interface=pi.default, i=size)


def test_invalid_signatures_are_rejected():
    for fun in (lambda *indices: indices, lambda *, i: i, lambda **axes: axes):
        with pytest.raises(TypeError, match="fixed positional signature"):
            loop(fun, interface=pi.default, i=2)
    with pytest.raises(ValueError, match="name is reserved"):
        loop(lambda interface: interface + 1, interface=pi.default)
    with pytest.raises(TypeError, match="expected a callable"):
        loop(42, interface=pi.default, i=2)


def test_existing_mapped_outputs_are_rejected():
    captured = pi.normal(0, 1)
    for fun in (lambda i: i, lambda i: captured):
        with pytest.raises(ValueError, match="every output must be created"):
            loop(fun, interface=pi.default, i=2)


def test_non_rv_outputs_including_trees_are_rejected_with_or_without_axes():
    captured = pi.normal(0, 1)
    Single = namedtuple("Single", ["x"])
    for value in (1, None, [], (), {}, [captured], (captured,), {"x": captured}, Single(captured)):
        with pytest.raises(TypeError, match="must return a single RV"):
            loop(lambda: value, interface=pi.default)
        with pytest.raises(TypeError, match="must return a single RV"):
            loop(lambda i: value, interface=pi.default, i=2)
    with pytest.raises(TypeError, match="must return a single RV"):
        loop(lambda i: (pi.normal(i, 1),), interface=pi.default, i=2)


def test_export_and_raw_ir_without_interface():
    assert functional.loop is loop
    assert "loop" in functional.__all__
    result = loop(lambda i, j: ir.RV(ir.Add(), i, j), interface=ir.NO_INTERFACE, j=3, i=2)
    assert result.interface is ir.NO_INTERFACE
    assert result.op == ir.VMap(ir.VMap(ir.Add(), (None, 0), 3), (0, None), 2)
    assert result.parents[0].op == ir.Constant([0, 1])
    assert result.parents[1].op == ir.Constant([0, 1, 2])
    np.testing.assert_array_equal(evaluate(result), [[0, 1, 2], [1, 2, 3]])


def test_raw_output_does_not_inherit_coordinate_interface():
    result = loop(lambda i: ir.RV(ir.Exp(), i), interface=pi.default, i=3)
    assert result.interface is ir.NO_INTERFACE
    np.testing.assert_allclose(evaluate(result), np.exp(np.arange(3)), rtol=1e-6)


def test_body_called_once_even_with_singleton_and_many_axes():
    seen = []

    def model(a, b, c, d, e, f, g):
        seen.append((a, b, c, d, e, f, g))
        return a + 2 * b + 4 * c + 8 * d + 16 * e + 32 * f + 64 * g

    result = loop(model, interface=pi.default, a=1, b=2, c=1, d=2, e=1, f=2, g=1)
    assert len(seen) == 1
    assert all(rv.shape == () for rv in seen[0])
    assert result.shape == (1, 2, 1, 2, 1, 2, 1)
    coordinates = np.indices(result.shape)
    expected = sum(coordinates[i] * 2**i for i in range(7))
    np.testing.assert_array_equal(evaluate(result), expected)


def test_unused_coordinates_and_new_constant_output():
    result = loop(lambda i, j: ir.RV(ir.Constant(7)), interface=ir.NO_INTERFACE, i=2, j=3)
    assert result.shape == (2, 3)
    assert result.interface is ir.NO_INTERFACE
    np.testing.assert_array_equal(evaluate(result), np.full((2, 3), 7))


def test_positional_only_callback_and_axis_named_fun():
    def model(fun, j, /):
        return fun * 10 + j

    result = loop(model, interface=pi.default, j=3, fun=2)
    np.testing.assert_array_equal(evaluate(result), [[0, 1, 2], [10, 11, 12]])


def test_bound_method_and_positional_partial():
    from functools import partial

    class Model:
        def body(self, offset, i):
            return offset + i

    model = Model()
    result = loop(partial(model.body, pi.constant(4)), interface=pi.default, i=3)
    np.testing.assert_array_equal(evaluate(result), [4, 5, 6])


def test_current_configuration_applies_inside_body():
    configured = pi.Interface(broadcasting="off")
    with configured.overload(broadcasting="simple"):
        result = loop(lambda i: configured.add(i, [1, 2, 3]), interface=configured, i=2)
    assert result.shape == (2, 3)
    assert result.interface is configured
    assert configured._effective_config.broadcasting == "off"
    np.testing.assert_array_equal(evaluate(result), [[1, 2, 3], [2, 3, 4]])


def test_callback_error_does_not_poison_subsequent_loop():
    class ModelError(Exception):
        pass

    def broken(i):
        pi.normal(i, 1)
        raise ModelError("body failed")

    with pytest.raises(ModelError, match="body failed"):
        loop(broken, interface=pi.default, i=2)
    result = loop(lambda i: i + 1, interface=pi.default, i=3)
    np.testing.assert_array_equal(evaluate(result), [1, 2, 3])


def test_nested_loop_cannot_return_existing_outer_random_variable():
    def outer(i):
        offset = pi.normal(i, 1)
        return loop(lambda j: offset, interface=pi.default, j=3)

    with pytest.raises(ValueError, match="every output must be created"):
        loop(outer, interface=pi.default, i=2)


def test_custom_integer_size_and_numpy_integer_size():
    class Size:
        def __index__(self):
            return 2

    result = loop(lambda i, j: i * j, interface=pi.default, i=Size(), j=np.int32(3))
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(evaluate(result), [[0, 0, 0], [0, 1, 2]])
