"""Exact Scan/VMap wiring regressions adapted from the sfor tests."""

import jax
import numpy as np

from pangolin import interface as pi, ir, jax_backend
from pangolin.functional import loop


def test_unused_sequential_coordinate_disappears_from_op_and_parents():
    def model(prev, t):
        if t is None:
            return pi.constant(0.0)
        return pi.normal(prev[t - 1], 1)

    x = loop(
        model,
        interface=pi.default,
        t=[100],
    )
    assert x.op == ir.Scan(ir.Composite(1, (ir.Constant(1), ir.Normal()), ((), (0, 1))), 100, ())
    assert len(x.parents) == 1
    assert x.parents[0].op == ir.Constant(0.0)


def test_captured_parents_have_explicit_stable_wiring():
    offset = pi.constant(5)
    scale = pi.constant(6)

    def model(prev, t):
        if t is None:
            return pi.constant(0.0)
        return pi.normal(prev[t - 1] + offset, scale + t)

    x = loop(
        model,
        interface=pi.default,
        t=[3],
    )
    assert x.op == ir.Scan(
        ir.Composite(4, (ir.Add(), ir.Add(), ir.Normal()), ((0, 2), (3, 1), (4, 5))),
        3,
        (0, None, None),
    )
    assert len(x.parents) == 4
    assert x.parents[0].op == ir.Constant(0.0)
    assert x.parents[1].op == ir.Constant([0, 1, 2])
    assert x.parents[2] is offset
    assert x.parents[3] is scale


def test_captured_array_is_indexed_by_time_and_parallel_coordinate():
    scale = pi.constant([[0.5, 0.7], [0.9, 1.1], [1.3, 1.5]])

    def model(prev, t, i):
        if t is None:
            return pi.constant(0.0)
        return pi.normal(prev[t - 1, i], scale[t, i])

    x = loop(
        model,
        interface=pi.default,
        t=[3],
        i=2,
    )
    assert x.shape == (3, 2)
    assert x.op == ir.Scan(ir.VMap(ir.Normal(), (0, 0), 2), 3, (0,))
    assert len(x.parents) == 2
    assert x.parents[0].op == ir.VMap(ir.Constant(0.0), (), 2)
    assert x.parents[0].parents == ()
    assert x.parents[1] is scale


def test_multiple_parallel_axes_have_exact_nested_vmap_order():
    def model(prev, time, row, column):
        if time is None:
            return pi.constant(0.0)
        return pi.normal(prev[time - 1, row, column], 1)

    x = loop(
        model,
        interface=pi.default,
        column=4,
        time=[2],
        row=3,
    )
    assert x.shape == (2, 3, 4)
    assert x.op == ir.Scan(
        ir.Composite(
            1,
            (
                ir.Constant(1),
                ir.VMap(ir.VMap(ir.Normal(), (0, None), 4), (0, None), 3),
            ),
            ((), (0, 1)),
        ),
        2,
        (),
    )
    assert len(x.parents) == 1
    assert x.parents[0].op == ir.VMap(ir.VMap(ir.Constant(0.0), (), 4), (), 3)
    assert x.parents[0].parents == ()


def test_parallel_components_can_return_event_vectors():
    def model(prev, t, i):
        if t is None:
            return pi.constant(np.zeros(3))
        return pi.normal(prev[t - 1, i, :], 1)

    x = loop(
        model,
        interface=pi.default,
        t=[4],
        i=2,
    )
    assert x.shape == (4, 2, 3)
    assert x.op == ir.Scan(
        ir.Composite(
            1,
            (
                ir.Constant(1),
                ir.VMap(ir.VMap(ir.Normal(), (0, None), 3), (0, None), 2),
            ),
            ((), (0, 1)),
        ),
        4,
        (),
    )
    assert len(x.parents) == 1
    assert x.parents[0].op == ir.VMap(ir.Constant(np.zeros(3)), (), 2)
    assert x.parents[0].parents == ()


def test_explicit_inner_loop_and_parallel_axis_have_same_scan_op():
    def model(prev, t, i):
        if t is None:
            return pi.constant(0.0)
        return pi.normal(prev[t - 1, i], 1)

    compact = loop(
        model,
        interface=pi.default,
        t=[3],
        i=2,
    )

    def body(prev, t):
        if t is None:
            return pi.constant(np.zeros(2))
        return loop(lambda i: pi.normal(prev[t - 1, i], 1), i=2, interface=pi.default)

    explicit = loop(body, interface=pi.default, t=[3])
    expected = ir.Scan(
        ir.Composite(
            1,
            (ir.Constant(1), ir.VMap(ir.Normal(), (0, None), 2)),
            ((), (0, 1)),
        ),
        3,
        (),
    )
    assert compact.op == explicit.op == expected
    assert compact.shape == explicit.shape == (3, 2)


def test_positional_only_callbacks_and_numpy_integer_sizes():
    def initial(i, /):
        return i + 10

    def body(prev, t, i, /):
        if t is None:
            return initial(i)
        return prev[t - 1, i] + t + i

    x = loop(
        body,
        interface=pi.default,
        i=np.int32(2),
        t=[np.int64(3)],
    )
    assert x.shape == (3, 2)
    assert isinstance(x.op, ir.Scan)
    assert type(x.op.length) is int
    assert x.op.length == 3
    sample = jax.jit(jax_backend.ancestor_sampler(x))(jax.random.key(0))
    np.testing.assert_array_equal(sample, [[10, 12], [11, 14], [13, 17]])
