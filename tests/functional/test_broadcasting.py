"""Check broadcasting shape rules, numerical axis routing, and RV interfaces."""

import inspect
from math import prod

import numpy as np
import pytest

from pangolin import dag, functional, ir

COMBINATORS = (
    functional.broadcast_off,
    functional.broadcast_simple,
    functional.broadcast_rank,
    functional.broadcast_numpy,
)

CASES = [
    (((), ()), 0),
    (((3,), ()), 1),
    (((), (2, 3)), 1),
    (((1,), (1,)), 1),
    (((2, 3), (2, 3)), 1),
    (((3,), (2, 3)), 2),
    (((2, 3), (3,)), 2),
    (((1,), (2, 1)), 2),
    (((2, 1, 3), (1, 3)), 2),
    (((2, 1), (3,)), 3),
    (((1, 3), (2, 1)), 3),
    (((1,), (2, 3)), 3),
    (((1, 3, 1), (2, 1, 4)), 3),
    (((2, 1, 4), (1, 3, 1)), 3),
    (((1, 2, 1, 3), (2, 1, 4, 1)), 3),
    (((2,), (3,)), None),
    (((2, 3), (2,)), None),
    (((2, 3), (3, 2)), None),
]


def subtract(x: ir.RV, y: ir.RV) -> ir.RV:
    return ir.RV(ir.Sub(), x, y)


def evaluate_op(op, values):
    if isinstance(op, ir.VMap):
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
    if isinstance(op, ir.Index):
        array, *indices = values
        return array[tuple(indices)]
    assert op == ir.Sub()
    assert all(value.shape == () for value in values)
    return values[0] - values[1]


def evaluate(rv):
    if isinstance(rv.op, ir.Constant):
        return rv.op.value
    return evaluate_op(rv.op, tuple(evaluate(parent) for parent in rv.parents))


@pytest.mark.parametrize("shapes,lowest_mode", CASES)
@pytest.mark.parametrize("mode", range(4))
def test_shape_rules_and_numerical_routing(shapes, lowest_mode, mode):
    arrays = tuple(
        np.arange(prod(shape)).reshape(shape) * (i + 2) + 100 * i for i, shape in enumerate(shapes)
    )
    args = tuple(ir.RV(ir.Constant(array)) for array in arrays)
    broadcasted = COMBINATORS[mode](subtract)
    if lowest_mode is None or mode < lowest_mode:
        with pytest.raises(ValueError, match="Can't broadcast shapes"):
            broadcasted(*args)
    else:
        result = broadcasted(*args)
        assert result.shape == np.broadcast_shapes(*shapes)
        np.testing.assert_array_equal(evaluate(result), arrays[0] - arrays[1])
        assert all(arg in dag.upstream_nodes(result) for arg in args)


@pytest.mark.parametrize("combinator", COMBINATORS)
def test_keywords_defaults_and_signature(combinator):
    default_scale = ir.RV(ir.Constant(3))

    def normal(mean, scale=default_scale):
        return ir.RV(ir.Normal(), mean, scale)

    mean = ir.RV(ir.Constant(2))
    scale = ir.RV(ir.Constant(4))
    broadcasted = combinator(normal)
    assert inspect.signature(broadcasted) == inspect.signature(normal)
    assert broadcasted(mean=mean).parents == (mean, default_scale)
    assert broadcasted(scale=scale, mean=mean).parents == (mean, scale)
    with pytest.raises(TypeError):
        broadcasted()


@pytest.mark.parametrize("combinator", COMBINATORS)
def test_rejects_non_rv_arguments(combinator):
    with pytest.raises(TypeError, match="arguments must be RVs"):
        combinator(subtract)(ir.RV(ir.Constant(1)), 2)


@pytest.mark.parametrize("combinator", COMBINATORS)
def test_keyword_only_signature(combinator):
    def fun(x, *, y):
        return subtract(x, y)

    x, y = ir.RV(ir.Constant(3)), ir.RV(ir.Constant(2))
    assert combinator(fun)(x, y=y).parents == (x, y)


@pytest.mark.parametrize("combinator", COMBINATORS)
def test_rejects_non_rv_keyword_arguments(combinator):
    with pytest.raises(TypeError, match="arguments must be RVs"):
        combinator(subtract)(ir.RV(ir.Constant(1)), y=2)


@pytest.mark.parametrize("combinator", COMBINATORS)
def test_python_defaults_are_not_broadcast_inputs(combinator):
    captured = ir.RV(ir.Constant([4, 5]))

    def fun(x, operation="sub", parameter=captured):
        assert operation == "sub"
        assert parameter is captured
        return subtract(x, ir.RV(ir.Constant(1)))

    x = ir.RV(ir.Constant(3))
    np.testing.assert_array_equal(evaluate(combinator(fun)(x)), 2)


@pytest.mark.parametrize("combinator", COMBINATORS[1:])
def test_omitted_default_matches_captured_value(combinator):
    default = ir.RV(ir.Constant(2))
    x = ir.RV(ir.Constant([3, 4, 5]))

    def fun(x, y=default):
        assert y is default
        return subtract(x, y)

    result = combinator(fun)(x)
    assert result.parents == (x, default)
    assert result.op == ir.VMap(ir.Sub(), (0, None), 3)


@pytest.mark.parametrize("shapes,lowest_mode", CASES)
@pytest.mark.parametrize("mode", range(4))
def test_keyword_routing_matches_positional(shapes, lowest_mode, mode):
    arrays = tuple(np.arange(prod(shape)).reshape(shape) for shape in shapes)
    x, y = (ir.RV(ir.Constant(array)) for array in arrays)
    broadcasted = COMBINATORS[mode](subtract)
    for args, kwargs in (((x,), {"y": y}), ((), {"y": y, "x": x})):
        if lowest_mode is None or mode < lowest_mode:
            with pytest.raises(ValueError, match="Can't broadcast shapes"):
                broadcasted(*args, **kwargs)
        else:
            np.testing.assert_array_equal(
                evaluate(broadcasted(*args, **kwargs)), arrays[0] - arrays[1]
            )


@pytest.mark.parametrize("combinator", COMBINATORS[1:])
def test_variadic_keyword_arguments(combinator):
    def fun(**kwargs):
        return subtract(kwargs["x"], kwargs["y"])

    x, y = ir.RV(ir.Constant([4, 5, 6])), ir.RV(ir.Constant(2))
    np.testing.assert_array_equal(evaluate(combinator(fun)(y=y, x=x)), [2, 3, 4])


@pytest.mark.parametrize("combinator", COMBINATORS)
def test_invalid_calls_raise_from_function(combinator):
    x = ir.RV(ir.Constant(1))
    broadcasted = combinator(subtract)
    with pytest.raises(TypeError):
        broadcasted(x, x, y=x)
    with pytest.raises(TypeError):
        broadcasted(x, z=x)


@pytest.mark.parametrize("combinator", COMBINATORS)
def test_zero_arguments(combinator):
    result = combinator(lambda: ir.RV(ir.Constant(1)))()
    assert result.op == ir.Constant(1)


@pytest.mark.parametrize("combinator", COMBINATORS[1:])
def test_variadic_function_and_three_arguments(combinator):
    def fun(*args):
        return subtract(subtract(args[0], args[1]), args[2])

    arrays = (np.arange(6).reshape(2, 3), np.array(10), np.full((2, 3), 4))
    result = combinator(fun)(*(ir.RV(ir.Constant(array)) for array in arrays))
    np.testing.assert_array_equal(evaluate(result), arrays[0] - arrays[1] - arrays[2])


@pytest.mark.parametrize("combinator", COMBINATORS[1:])
def test_unary_multidimensional_input(combinator):
    x = ir.RV(ir.Constant(np.zeros((2, 3, 1))))
    result = combinator(lambda x: ir.RV(ir.Exp(), x))(x)
    assert result.shape == x.shape
    assert result.parents == (x,)


@pytest.mark.parametrize("combinator", COMBINATORS[1:])
def test_captured_rv_is_shared_across_broadcast(combinator):
    rate = ir.RV(ir.Constant(1))
    captured = ir.RV(ir.Exponential(), rate)
    x = ir.RV(ir.Constant([1, 2, 3]))
    result = combinator(lambda x: subtract(x, captured))(x)
    assert result.parents == (x, captured)
    assert result.op == ir.VMap(ir.Sub(), (0, None), 3)


def test_numpy_squeezing_preserves_input_and_selected_output_interfaces():
    first, second, output = ir.RVInterface(), ir.RVInterface(), ir.RVInterface()
    x = ir.RV(ir.Constant([[1], [2]]), interface=first)
    y = ir.RV(ir.Constant([[3, 4, 5]]), interface=second)

    def fun(a, b):
        assert a.interface is first
        assert b.interface is second
        return ir.RV(ir.Sub(), a, b, interface=output)

    result = functional.broadcast_numpy(fun)(x, y)
    assert result.interface is output
    assert result.shape == (2, 3)
    assert x in dag.upstream_nodes(result)
    assert y in dag.upstream_nodes(result)
