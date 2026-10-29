"""Check indexing syntax, shape rules, numerical routing, and interface ownership."""

import numpy as np
import pytest

from pangolin import dag, functional, ir
from pangolin import interface as pi
from pangolin.interface import Interface, convert_index, eliminate_ellipses


def evaluate_op(op, values):
    if isinstance(op, ir.Index):
        result, *indices = values
        axis = 0
        for index in indices:
            result = np.take(result, index, axis=axis)
            axis += index.ndim
        return result
    assert isinstance(op, ir.VMap)
    return np.stack(
        [
            evaluate_op(
                op.base_op,
                tuple(
                    value if axis is None else np.take(value, i, axis=axis)
                    for value, axis in zip(values, op.in_axes, strict=True)
                ),
            )
            for i in range(op.axis_size)
        ]
    )


def evaluate(rv):
    if isinstance(rv.op, ir.Constant):
        return rv.op.value
    return evaluate_op(rv.op, tuple(evaluate(parent) for parent in rv.parents))


@pytest.mark.parametrize("mode", ["off", "simple", "rank", "numpy"])
@pytest.mark.parametrize(
    "indices",
    [
        (1, 2),
        (-1, -2),
        (slice(None), slice(None)),
        (slice(None, None, -1), slice(1, 4, 2)),
        ([3, 0], [2, 1, 2]),
        ([[3, 0], [1, 2]], [[0], [2], [3]]),
        (slice(None), [2, 0]),
        (2, ...),
        (..., 1),
        (...,),
    ],
)
def test_orthogonal_indexing(mode, indices):
    interface = Interface(broadcasting=mode)
    array = np.arange(16).reshape(4, 4)
    x = interface.constant(array)
    explicit = interface.oindex(x, *indices)
    syntax = x[indices]
    normalized = eliminate_ellipses(2, indices)
    integer_indices = tuple(
        np.arange(size)[index] if isinstance(index, slice) else np.asarray(index)
        for size, index in zip(array.shape, normalized, strict=True)
    )
    expected = array[integer_indices[0]]
    expected = np.take(expected, integer_indices[1], axis=integer_indices[0].ndim)
    for result in (explicit, syntax, interface.getitem(x, *indices)):
        op = result.op
        for i, index in enumerate(integer_indices):
            for size in index.shape:
                assert isinstance(op, ir.VMap)
                assert op.axis_size == size
                assert op.in_axes == tuple(0 if j == i + 1 else None for j in range(3))
                op = op.base_op
        assert op == ir.Index()
        assert result.shape == expected.shape
        assert result.parents[0] is x
        assert result.interface is interface
        np.testing.assert_array_equal(evaluate(result), expected)


def test_orthogonal_indexing_multiple_index_dimensions():
    x = pi.constant(np.arange(60).reshape(3, 4, 5))
    rows = pi.constant([[2, 0, 1], [0, 2, 0]])
    columns = pi.constant([[4], [1], [0], [3]])
    result = x[rows, 2, columns]
    assert result.op == ir.VMap(
        ir.VMap(
            ir.VMap(
                ir.VMap(ir.Index(), (None, None, None, 0), 1),
                (None, None, None, 0),
                4,
            ),
            (None, 0, None, None),
            3,
        ),
        (None, 0, None, None),
        2,
    )
    assert result.shape == (2, 3, 4, 1)
    assert result.parents[0] is x
    assert result.parents[1] is rows
    assert result.parents[3] is columns
    expected = x.op.value[rows.op.value, 2][..., columns.op.value]
    np.testing.assert_array_equal(evaluate(result), expected)


@pytest.mark.parametrize(
    "kinds,shape",
    [
        ("iii", ()),
        ("vii", (5,)),
        ("ivi", (5,)),
        ("iiv", (5,)),
        ("mii", (3, 2)),
        ("imi", (3, 2)),
        ("iim", (3, 2)),
        ("vmi", (5, 3, 2)),
        ("vim", (5, 3, 2)),
        ("mvi", (3, 2, 5)),
        ("miv", (3, 2, 5)),
        ("mmi", (3, 2, 3, 2)),
        ("mim", (3, 2, 3, 2)),
        ("sii", (2,)),
        ("vis", (5, 4)),
        ("ivs", (5, 4)),
        ("siv", (2, 5)),
        ("msi", (3, 2, 3)),
        ("ims", (3, 2, 4)),
        ("sim", (2, 3, 2)),
        ("vms", (5, 3, 2, 4)),
        ("vsm", (5, 3, 3, 2)),
        ("mvs", (3, 2, 5, 4)),
        ("msv", (3, 2, 3, 5)),
        ("mms", (3, 2, 3, 2, 4)),
        ("msm", (3, 2, 3, 3, 2)),
    ],
)
def test_orthogonal_indexing_shapes_and_values(kinds, shape):
    """Preserve mixed scalar, vector, matrix, and slice coverage from the old IR helper."""
    array = np.arange(24).reshape(2, 3, 4)
    choices = {"i": 1, "v": [0, 1, 0, 1, 1], "m": [[0, 1], [0, 1], [0, 1]], "s": slice(None)}
    indices = tuple(choices[kind] for kind in kinds)
    result = pi.oindex(array, *indices)
    assert result.shape == shape
    expected = array
    axis = 0
    for size, index in zip(array.shape, indices, strict=True):
        values = np.arange(size)[index] if isinstance(index, slice) else np.asarray(index)
        expected = np.take(expected, values, axis=axis)
        axis += values.ndim
    np.testing.assert_array_equal(evaluate(result), expected)


@pytest.mark.parametrize("mode", ["off", "simple", "rank", "numpy"])
@pytest.mark.parametrize(
    "indices,allowed",
    [
        ((1, 2), {"off", "simple", "rank", "numpy"}),
        (([3, 1], 2), {"simple", "rank", "numpy"}),
        (([3, 1], [0, 2]), {"simple", "rank", "numpy"}),
        ((((3, 1), (0, 2)), (0, 2)), {"rank", "numpy"}),
        (([[3], [1]], [0, 1, 2]), {"numpy"}),
        (([0, 1], [0, 1, 2]), set()),
    ],
)
def test_vectorized_indexing(mode, indices, allowed):
    interface = Interface(broadcasting=mode)
    array = np.arange(16).reshape(4, 4)
    x = interface.constant(array)
    if mode not in allowed:
        with pytest.raises(ValueError, match="Can't broadcast shapes"):
            interface.vindex(x, *indices)
        with pytest.raises(ValueError, match="Can't broadcast shapes"):
            x.vindex[indices]
    else:
        expected = array[tuple(np.asarray(index) for index in indices)]
        for result in (interface.vindex(x, *indices), x.vindex[indices]):
            assert result.interface is interface
            assert x in dag.upstream_nodes(result)
            np.testing.assert_array_equal(evaluate(result), expected)


def test_vectorized_slices_select_diagonal_and_antidiagonal():
    x = pi.constant(np.arange(16).reshape(4, 4))
    np.testing.assert_array_equal(evaluate(x.vindex[:, :]), [0, 5, 10, 15])
    np.testing.assert_array_equal(evaluate(x.vindex[:, ::-1]), [3, 6, 9, 12])
    np.testing.assert_array_equal(evaluate(x.vindex[...]), [0, 5, 10, 15])


@pytest.mark.parametrize("method", ["oindex", "vindex"])
@pytest.mark.parametrize("indices", [(1,), (1, 2, 3), (..., ...), (1, 2, ..., 3)])
def test_all_dimensions_must_be_indexed(method, indices):
    with pytest.raises(ValueError):
        getattr(pi, method)(np.arange(4).reshape(2, 2), *indices)


@pytest.mark.parametrize("method", ["oindex", "vindex"])
def test_slice_conversion_and_invalid_slices(method):
    result = getattr(pi, method)(np.arange(5), slice(-1, None, -2))
    np.testing.assert_array_equal(evaluate(result), [4, 2, 0])
    with pytest.raises(ValueError):
        getattr(pi, method)(np.arange(5), slice(None, None, 0))
    with pytest.raises(ValueError):
        getattr(pi, method)(np.arange(5), slice(2, 2))


def test_scalar_indexing_requires_explicit_ellipsis_for_vindex():
    x = pi.constant(3)
    for result in (pi.oindex(x), x[()], x[...], pi.vindex(x, ...), x.vindex[...]):
        assert result.op == ir.Index()
        assert result.parents == (x,)
        assert result.shape == ()
    with pytest.raises(TypeError, match="at least one index"):
        pi.vindex(x)
    with pytest.raises(TypeError, match="at least one index"):
        x.vindex[()]


def test_single_list_index_is_not_unpacked():
    x = pi.constant([1, 2, 3])
    np.testing.assert_array_equal(evaluate(x.vindex[[2, 0]]), [3, 1])
    np.testing.assert_array_equal(evaluate(x[[2, 0]]), [3, 1])


@pytest.mark.parametrize("method", ["oindex", "vindex"])
def test_called_interface_owns_output_but_not_existing_parents(method):
    interface, foreign = Interface(), Interface(broadcasting="off")
    x = foreign.constant([[1, 2], [3, 4]])
    index = ir.RV(ir.Constant([1, 0]))
    result = getattr(interface, method)(x, index, index)
    assert result.interface is interface
    assert x.interface is foreign
    assert index.interface is ir.NO_INTERFACE
    upstream = dag.upstream_nodes(result)
    assert x in upstream
    assert index in upstream
    assert len([rv for rv in upstream if rv is index]) == 1


def test_new_index_constants_use_called_interface():
    interface, foreign = Interface(), Interface(broadcasting="off")
    x = foreign.constant([[1, 2], [3, 4]])
    for result in (interface.oindex(x, slice(None), 0), interface.vindex(x, slice(None), 0)):
        assert result.interface is interface
        assert result.parents[0] is x
        assert all(parent.interface is interface for parent in result.parents[1:])


def test_random_array_and_indices_keep_graph_identity():
    interface = Interface()
    x = interface.multi_normal([0, 0], np.eye(2))
    index = interface.categorical([0.5, 0.5])
    for result in (x[index], x.vindex[index]):
        assert result.op == ir.Index()
        assert result.parents == (x, index)
        assert not result.op.random


def test_proxy_defers_dispatch_and_reads_config_at_use():
    interface = Interface(broadcasting="off")
    x = interface.constant([[1, 2], [3, 4]])
    proxy = x.vindex
    with pytest.raises(ValueError, match="broadcasting=OFF"):
        proxy[:, :]
    with interface.overload(broadcasting="simple"):
        np.testing.assert_array_equal(evaluate(proxy[:, :]), [1, 4])
    with pytest.raises(ValueError, match="broadcasting=OFF"):
        proxy[:, :]


def test_index_conversion_preserves_existing_rvs():
    x = ir.RV(ir.Constant([2, 0]))
    assert convert_index(pi.default, 3, x) is x


def test_indexing_composes_with_vmap_and_composite():
    x = pi.constant(np.arange(6).reshape(2, 3))
    for combinator in (functional.vmap, functional.composite):
        result = combinator(lambda a: a[..., 0])(x)
        assert result.interface is pi.default
        assert x in dag.upstream_nodes(result)


def test_indexing_default_exports():
    assert {"oindex", "vindex"} <= set(pi.__all__)
    assert pi.oindex == pi.default.oindex
    assert pi.vindex == pi.default.vindex
