"""Systematic acceptance, structure, and numerical tests for scalar-function broadcasting."""

from __future__ import annotations

from pangolin import interface as pi, ir


from collections.abc import Callable
from math import prod

import numpy as np
import pytest

from pangolin import dag
from pangolin.functional.tracing import AbstractOp

THREE_ARGUMENT_OP = ir.Composite(3, (ir.Sub(), ir.Mul()), ((0, 1), (3, 2)))
BROADCASTING_MODES = ("off", "simple", "rank", "numpy")

BROADCAST_CASES = [
    # Scalars
    [(), "off"],
    [(), (), "off"],
    [(), (), (), "off"],
    # Unary arrays
    [(1,), "simple"],
    [(2,), "simple"],
    [(1, 1), "simple"],
    [(2, 3), "simple"],
    [(1, 2, 3), "simple"],
    [(2, 1, 3, 1), "simple"],
    [(1, 1, 1, 1, 1, 1), "simple"],
    [(2, 3, 1, 2, 3, 1, 2, 3), "simple"],
    # Equal array shapes and scalars
    [(1,), (), "simple"],
    [(), (1,), "simple"],
    [(2,), (2,), "simple"],
    [(2, 3), (), "simple"],
    [(), (2, 3), "simple"],
    [(2, 3), (2, 3), "simple"],
    [(2, 1, 3, 1), (2, 1, 3, 1), "simple"],
    [(2,), (2,), (2,), "simple"],
    [(), (2,), (2,), "simple"],
    [(2,), (), (2,), "simple"],
    [(2,), (2,), (), "simple"],
    [(), (2, 3), (2, 3), "simple"],
    [(2, 3), (), (2, 3), "simple"],
    [(2, 3), (2, 3), (), "simple"],
    [(2, 1, 3, 1), (), (2, 1, 3, 1), "simple"],
    # Rank promotion
    [(3,), (2, 3), "rank"],
    [(2, 3), (3,), "rank"],
    [(), (3,), (2, 3), "rank"],
    [(3,), (), (2, 3), "rank"],
    [(3,), (2, 3), (), "rank"],
    [(1,), (3, 1), "rank"],
    [(2, 1), (3, 2, 1), "rank"],
    [(1, 3), (2, 1, 3), "rank"],
    [(4, 3), (2, 4, 3), "rank"],
    [(3,), (4, 3), (2, 4, 3), "rank"],
    [(2, 4, 3), (3,), (4, 3), "rank"],
    [(4, 3), (2, 4, 3), (3,), "rank"],
    [(1,), (2, 1), (3, 2, 1), "rank"],
    [(3, 1, 2), (2, 1, 3, 1, 2), "rank"],
    [(2, 1, 3, 1, 2), (3, 1, 2), "rank"],
    [(), (2, 1), (3, 2, 1), "rank"],
    [(2, 1), (), (3, 2, 1), "rank"],
    [(2, 1), (3, 2, 1), (), "rank"],
    [(3,), (2, 3), (4, 2, 3), "rank"],
    [(2, 3), (4, 2, 3), (2, 3), "rank"],
    [(2, 3, 1), (3, 2, 3, 1), (4, 3, 2, 3, 1), "rank"],
    [(3, 1, 2), (2, 3, 1, 2), (4, 2, 3, 1, 2), "rank"],
    [(2, 3, 1, 2), (4, 2, 3, 1, 2), "rank"],
    [(2, 3, 1, 2, 3, 1), (4, 2, 3, 1, 2, 3, 1), "rank"],
    # NumPy singleton expansion
    [(1,), (5,), "numpy"],
    [(5,), (1,), "numpy"],
    [(1, 3), (5, 3), "numpy"],
    [(5, 3), (1, 3), "numpy"],
    [(2, 1), (2, 4), "numpy"],
    [(2, 4), (2, 1), "numpy"],
    [(2, 3, 1), (2, 3, 4), "numpy"],
    [(2, 3, 4), (2, 3, 1), "numpy"],
    [(1, 3, 1), (2, 1, 4), "numpy"],
    [(2, 1, 4), (1, 3, 1), "numpy"],
    [(1,), (2, 3), "numpy"],
    [(2, 3), (1,), "numpy"],
    [(2, 1), (3,), "numpy"],
    [(3,), (2, 1), "numpy"],
    [(1, 2), (3, 1), "numpy"],
    [(3, 1), (1, 2), "numpy"],
    [(3,), (2, 1, 1), "numpy"],
    [(2, 1, 1), (3,), "numpy"],
    [(1, 1, 3), (2, 4, 1), "numpy"],
    [(2, 1, 3), (1, 4, 1), "numpy"],
    [(1, 3, 1), (2, 1, 4), (2, 3, 1), "numpy"],
    [(2, 1, 1), (1, 3, 1), (1, 1, 4), "numpy"],
    [(1, 3), (5, 1), (5, 3), "numpy"],
    [(1,), (3,), (), "numpy"],
    [(), (1,), (3,), "numpy"],
    [(3,), (), (1,), "numpy"],
    [(3,), (1, 4, 1), (2, 1, 3), "numpy"],
    [(1, 2, 1, 4), (3, 1, 5, 1), "numpy"],
    [(2, 1, 3, 1, 4), (1, 5, 1, 6, 1), "numpy"],
    [(1, 3, 1, 5, 1), (2, 1, 4, 1, 6), "numpy"],
    [(2, 1, 1, 5), (1, 3, 1, 1), (1, 1, 4, 5), "numpy"],
    [(1, 3, 1, 5, 1, 7), (2, 1, 4, 1, 6, 1), "numpy"],
    # Incompatible shapes
    [(2,), (3,), None],
    [(2, 3), (2, 4), None],
    [(2, 3), (3, 2), None],
    [(2, 3), (4,), None],
    [(2, 3), (2,), None],
    [(2, 1, 3), (2, 2, 4), None],
    [(2, 3, 4), (3, 5, 4), None],
    [(1, 2, 3), (4, 3), None],
    [(2, 3), (4, 3, 3), None],
    [(2,), (3,), (), None],
    [(1,), (2,), (3,), None],
    [(2, 1), (1, 3), (4, 3), None],
    [(2, 3), (1, 3), (2, 4), None],
    [(2, 1, 3), (1, 4, 1), (2, 5, 3), None],
    [(2, 3, 4, 5), (2, 3, 6, 5), None],
    [(1, 2, 1, 4), (3, 1, 5, 1), (3, 2, 6, 4), None],
]


def three_argument_scalar_function(a: ir.RV, b: ir.RV, c: ir.RV) -> ir.RV:
    return ir.RV(THREE_ARGUMENT_OP, a, b, c, interface=pi.default)


def three_argument_function(a: ir.RV, b: ir.RV, c: ir.RV) -> ir.RV:
    mode = pi.default._effective_config.broadcasting
    return pi.broadcasting_combinators[mode](three_argument_scalar_function)(a, b, c)


def case_id(case):
    return f"{case[:-1]}-{case[-1]}"


def rv_with_shape(shape: ir.Shape) -> ir.RV:
    """Create a symbolic input without allocating an array of the requested shape."""

    return ir.RV(AbstractOp(shape), interface=pi.default)


def call_scalar_function(args: tuple[ir.RV, ...]) -> ir.RV:
    """Call a deterministic scalar function matching the requested arity."""

    if len(args) == 1:
        return pi.abs(args[0])
    if len(args) == 2:
        return pi.sub(args[0], args[1])
    if len(args) == 3:
        return three_argument_function(args[0], args[1], args[2])
    raise AssertionError(f"unsupported arity: {len(args)}")


def expected_base_op(arity: int) -> ir.Op:
    """Return the scalar Op underlying the function selected by arity."""

    return {1: ir.Abs(), 2: ir.Sub(), 3: THREE_ARGUMENT_OP}[arity]


def unwrap_vmaps(op: ir.Op) -> tuple[list[ir.VMap], ir.Op]:
    """Separate a stack of `VMap` operations from its scalar base operation."""

    vmaps = []
    while isinstance(op, ir.VMap):
        vmaps.append(op)
        op = op.base_op
    return vmaps, op


def numpy_function(arity: int) -> Callable[..., np.ndarray]:
    """Return the NumPy equivalent of the scalar function selected by arity."""

    if arity == 1:
        return np.abs
    if arity == 2:
        return np.subtract
    if arity == 3:
        return lambda a, b, c: (a - b) * c
    raise AssertionError(f"unsupported arity: {arity}")


def interpret_broadcast_op(
    op: ir.Op,
    args: tuple[np.ndarray, ...],
    base_op: ir.Op,
    scalar_function: Callable[..., np.ndarray],
) -> np.ndarray:
    """Execute `VMap` and `Index` operations, delegating scalar arithmetic to NumPy."""

    if isinstance(op, ir.VMap):
        assert op.axis_size is not None
        outputs = []
        for index in range(op.axis_size):
            sliced_args = tuple(
                arg if axis is None else np.take(arg, index, axis=axis)
                for arg, axis in zip(args, op.in_axes, strict=True)
            )
            outputs.append(
                interpret_broadcast_op(op.base_op, sliced_args, base_op, scalar_function)
            )
        return np.stack(outputs)

    if isinstance(op, ir.Index):
        array, *indices = args
        return array[tuple(indices)]

    assert op == base_op
    assert all(arg.shape == () for arg in args)
    return scalar_function(*args)


def array_with_shape(shape: ir.Shape, argument: int) -> np.ndarray:
    """Create distinct integer values so incorrect argument or axis routing is visible."""

    values = np.arange(prod(shape), dtype=np.int64).reshape(shape)
    return values * (argument + 2) + 1000 * (argument + 1)


def interpret_broadcast_graph(
    rv: ir.RV,
    input_values: dict[ir.RV, np.ndarray],
    base_op: ir.Op,
    scalar_function: Callable[..., np.ndarray],
) -> np.ndarray:
    """Evaluate the small IR subset produced by deterministic scalar broadcasting."""

    if rv in input_values:
        return input_values[rv]
    if isinstance(rv.op, ir.Constant):
        return rv.op.value

    parent_values = tuple(
        interpret_broadcast_graph(parent, input_values, base_op, scalar_function)
        for parent in rv.parents
    )
    return interpret_broadcast_op(rv.op, parent_values, base_op, scalar_function)


@pytest.mark.parametrize("case", BROADCAST_CASES, ids=case_id)
def test_broadcasting_case(case):
    *shapes, lowest_mode = case
    shapes = tuple(shapes)
    if lowest_mode is None:
        accepted_modes = ()
    else:
        accepted_modes = BROADCASTING_MODES[BROADCASTING_MODES.index(lowest_mode) :]

    args = tuple(rv_with_shape(shape) for shape in shapes)
    arrays = tuple(array_with_shape(shape, argument) for argument, shape in enumerate(shapes))
    base_op = expected_base_op(len(args))
    scalar_function = numpy_function(len(args))
    input_values = {arg: array for arg, array in zip(args, arrays, strict=True)}
    observed_modes = []

    for mode in BROADCASTING_MODES:
        with pi.overload(broadcasting=mode):
            try:
                result = call_scalar_function(args)
            except ValueError:
                result = None

        observed_acceptance = result is not None
        assert observed_acceptance == (mode in accepted_modes)
        if observed_acceptance:
            observed_modes.append(mode)

        if result is None:
            continue

        expected_shape = np.broadcast_shapes(*shapes)
        assert result.shape == expected_shape
        assert len(result.parents) == len(args)
        for actual_parent, input_rv in zip(result.parents, args, strict=True):
            if mode == "numpy":
                assert input_rv in dag.upstream_nodes(actual_parent)
            else:
                assert actual_parent is input_rv

        vmaps, actual_base_op = unwrap_vmaps(result.op)
        assert actual_base_op == base_op
        assert len(vmaps) == len(expected_shape)

        actual = interpret_broadcast_graph(result, input_values, base_op, scalar_function)
        expected = scalar_function(*arrays)
        np.testing.assert_array_equal(actual, expected)

    assert tuple(observed_modes) == accepted_modes
