from pangolin import interface as pi, ir
from pangolin.ir import RV


import pytest

from pangolin.interface import add
from pangolin.interface import constant
from pangolin.interface import exp
from pangolin.interface import normal
from pangolin.functional import compositing
from pangolin.functional.compositing import composite
from pangolin.functional.compositing import composite_from_trace


def test_public_api_matches_compositing():
    assert compositing.__all__ == ["composite"]


def test_composite_passes_arguments_through_and_calls_function_once():
    calls = []
    value = [1.0, 2.0]

    def fun(x):
        calls.append(x)
        return exp(constant(x))

    result = composite(fun)(value)

    assert calls == [value]
    assert calls[0] is value
    assert result.parents == ()
    assert result.op == ir.Composite(
        0,
        (ir.Constant([1.0, 2.0]), ir.VMap(ir.Exp(), (0,), 2)),
        ((), (0,)),
    )


def test_composite_bakes_constants_created_from_arguments():
    def fun(loc, scale):
        return normal(loc, scale)

    result = composite(fun)(2.0, 3.0)

    assert result.parents == ()
    assert result.op == ir.Composite(
        0,
        (ir.Constant(2.0), ir.Constant(3.0), ir.Normal()),
        ((), (), (0, 1)),
    )


def test_composite_keeps_preexisting_boundary_rvs_as_parents():
    captured = constant(10)

    def fun(explicit):
        return captured - explicit

    explicit = constant(3)
    result = composite(fun)(explicit)

    assert result.parents == (explicit, captured)
    assert result.op == ir.Composite(2, (ir.Sub(),), ((1, 0),))


def test_composite_orders_argument_rvs_before_captures_in_pytree_leaf_order():
    captured = constant(10)
    a = constant(1)
    b = constant(2)

    def fun(values):
        return captured + values["b"] + values["a"]

    result = composite(fun)({"b": b, "a": a})

    assert result.parents == (a, b, captured)
    assert result.op == ir.Composite(
        3,
        (ir.Add(), ir.Add()),
        ((2, 1), (3, 0)),
    )


def test_composite_delegates_coercion_to_interface_operations():
    def fun(a, b):
        return add(a, b)

    result = composite(fun)([1, 2], [3, 4])

    assert result.shape == (2,)
    assert result.parents == ()
    assert result.op == ir.Composite(
        0,
        (ir.Constant([1, 2]), ir.Constant([3, 4]), ir.VMap(ir.Add(), (0, 0), 2)),
        ((), (), (0, 1)),
    )


def test_composite_does_not_make_an_invalid_call_valid():
    def fun(a, b):
        c = a + b
        return a / c

    with pytest.raises(TypeError):
        composite(fun)([1, 2, 3], [4, 5, 6])


def test_composite_excludes_new_nodes_that_do_not_contribute():
    def fun(x):
        exp(x)
        return normal(x, 1)

    x = constant(2)
    result = composite(fun)(x)

    assert result.parents == (x,)
    assert result.op == ir.Composite(
        1,
        (ir.Constant(1), ir.Normal()),
        ((), (0, 1)),
    )


def test_composite_requires_a_new_rv_output():
    x = constant(2)

    with pytest.raises(ValueError, match="every output must be created"):
        composite(lambda value: value)(x)


def test_composite_requires_an_rv_output():
    with pytest.raises(TypeError, match="found int"):
        composite(lambda x: x + 1)(2)


def test_composite_requires_exactly_one_rv_rather_than_an_rv_pytree():
    with pytest.raises(TypeError, match="exactly one newly created RV"):
        composite(lambda: [constant(1), constant(2)])()


def test_composite_from_trace_uses_the_supplied_parent_order():
    first = constant(1)
    second = constant(2)
    output = RV(ir.Sub(), first, second, interface=pi.default)

    op = composite_from_trace(output, [output], [second, first])

    assert op == ir.Composite(2, (ir.Sub(),), ((1, 0),))


@pytest.mark.parametrize("simplify", [False, True])
def test_composite_from_trace_simplifies_single_op(simplify):
    parent = constant(1)
    output = RV(ir.Exp(), parent)
    op = composite_from_trace(output, [output], [parent], simplify=simplify)
    expected = ir.Exp() if simplify else ir.Composite(1, (ir.Exp(),), ((0,),))
    assert op == expected


@pytest.mark.parametrize("simplify", [False, True])
def test_composite_from_trace_simplifies_existing_output(simplify):
    output = constant(1)
    op = composite_from_trace(output, [], [output], allow_existing_output=True, simplify=simplify)
    expected = ir.Identity() if simplify else ir.Composite(1, (ir.Identity(),), ((0,),))
    assert op == expected


def test_composite_from_trace_retains_unused_parents():
    used = constant(1)
    unused = constant(2)
    output = RV(ir.Exp(), used, interface=pi.default)

    op = composite_from_trace(output, [output], [unused, used])

    assert op == ir.Composite(2, (ir.Exp(),), ((1,),))


def test_composite_from_trace_rejects_a_missing_parent():
    parent = constant(1)
    output = RV(ir.Exp(), parent, interface=pi.default)

    with pytest.raises(ValueError, match="include every external parent"):
        composite_from_trace(output, [output], [])


def test_composite_from_trace_can_route_an_existing_output_through_identity():
    first = constant(1)
    output = constant(2)

    op = composite_from_trace(output, [], [first, output], allow_existing_output=True)

    assert op == ir.Composite(2, (ir.Identity(),), ((1,),))


def test_composite_from_trace_rejects_an_existing_output_by_default():
    output = constant(1)

    with pytest.raises(ValueError, match="final generated RV"):
        composite_from_trace(output, [], [output])


def test_composite_from_trace_requires_an_existing_output_to_be_a_parent():
    output = constant(1)

    with pytest.raises(ValueError, match="included in parents"):
        composite_from_trace(output, [], [], allow_existing_output=True)
