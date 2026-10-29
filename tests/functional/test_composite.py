import pytest

from pangolin import functional, ir
from pangolin.functional import compositing
from pangolin.functional.compositing import composite, composite_from_trace


@pytest.mark.parametrize("op", [ir.Add(), ir.Normal()])
def test_composite_simplifies_single_op(op):
    interface = ir.RVInterface()
    a = ir.RV(ir.Constant(1))
    b = ir.RV(ir.Constant(2))

    def fun(x: ir.RV, y: ir.RV) -> ir.RV:
        return ir.RV(op, x, y, interface=interface)

    result = composite(fun, simplify=True)(a, b)

    assert result.op is op
    assert result.parents == (a, b)
    assert result.interface is interface
    assert composite(fun)(a, b).op is op
    assert isinstance(composite(fun, simplify=False)(a, b).op, ir.Composite)


def test_composite_simplifies_constant():
    result = composite(lambda: ir.RV(ir.Constant(3)), simplify=True)()
    assert result.op == ir.Constant(3)
    assert result.parents == ()


def test_composite_does_not_simplify_reordered_inputs():
    a = ir.RV(ir.Constant(1))
    b = ir.RV(ir.Constant(2))
    result = composite(lambda x, y: ir.RV(ir.Sub(), y, x), simplify=True)(a, b)
    assert result.op == ir.Composite(2, (ir.Sub(),), ((1, 0),))
    assert result.parents == (a, b)


def test_composite_does_not_simplify_repeated_inputs():
    a = ir.RV(ir.Constant(1))
    result = composite(lambda x: ir.RV(ir.Add(), x, x), simplify=True)(a)
    assert result.op == ir.Composite(1, (ir.Add(),), ((0, 0),))
    assert result.parents == (a,)


def test_composite_does_not_simplify_multiple_ops():
    a = ir.RV(ir.Constant(1))
    result = composite(lambda x: ir.RV(ir.Exp(), ir.RV(ir.Log(), x)), simplify=True)(a)
    assert result.op == ir.Composite(1, (ir.Log(), ir.Exp()), ((0,), (1,)))


def test_public_api_contains_only_composite():
    assert compositing.__all__ == ["composite"]
    assert functional.composite is composite


def test_composite_compresses_a_plain_ir_graph():
    first = ir.RV(ir.Constant(1))
    second = ir.RV(ir.Constant(2))

    def fun(a: ir.RV, b: ir.RV) -> ir.RV:
        total = ir.RV(ir.Add(), a, b)
        return ir.RV(ir.Normal(), total, b)

    result = composite(fun)(first, second)

    assert type(result) is ir.RV
    assert result.parents == (first, second)
    assert result.op == ir.Composite(
        2,
        (ir.Add(), ir.Normal()),
        ((0, 1), (2, 1)),
    )


def test_composite_names_the_returned_function():
    def exponential(value: ir.RV) -> ir.RV:
        return ir.RV(ir.Exp(), value)

    composite_exponential = composite(exponential)

    assert getattr(composite_exponential, "__name__") == "composite_exponential"
    assert getattr(composite_exponential, "__qualname__") == "composite_exponential"


def test_composite_preserves_the_output_interface():
    interface = ir.RVInterface()
    value = ir.RV(ir.Constant(1), interface=interface)

    result = composite(lambda x: ir.RV(ir.Exp(), x, interface=interface))(value)

    assert type(result) is ir.RV
    assert result.interface is interface
    assert result.parents == (value,)


def test_composite_does_not_require_input_and_output_interfaces_to_match():
    input_interface = ir.RVInterface()
    output_interface = ir.RVInterface()
    value = ir.RV(ir.Constant(1), interface=input_interface)

    result = composite(lambda x: ir.RV(ir.Exp(), x, interface=output_interface))(value)

    assert result.interface is output_interface
    assert result.interface is not value.interface
    assert result.parents == (value,)


def test_composite_orders_arguments_before_captured_rvs():
    captured = ir.RV(ir.Constant(10))
    first = ir.RV(ir.Constant(1))
    second = ir.RV(ir.Constant(2))

    def fun(values: dict[str, ir.RV]) -> ir.RV:
        intermediate = ir.RV(ir.Add(), captured, values["second"])
        return ir.RV(ir.Add(), intermediate, values["first"])

    result = composite(fun)({"second": second, "first": first})

    assert result.parents == (first, second, captured)
    assert result.op == ir.Composite(
        3,
        (ir.Add(), ir.Add()),
        ((2, 1), (3, 0)),
    )


def test_composite_omits_new_nodes_that_do_not_contribute():
    value = ir.RV(ir.Constant(1))

    def fun(x: ir.RV) -> ir.RV:
        ir.RV(ir.Log(), x)
        return ir.RV(ir.Exp(), x)

    result = composite(fun, simplify=False)(value)

    assert result.op == ir.Composite(1, (ir.Exp(),), ((0,),))


def test_composite_passes_non_rv_arguments_through_unchanged():
    value = ir.RV(ir.Constant(1))
    configuration = {"operation": "log"}
    observed: list[object] = []

    def fun(x: ir.RV, options: dict[str, str]) -> ir.RV:
        observed.append(options)
        op = ir.Log() if options["operation"] == "log" else ir.Exp()
        return ir.RV(op, x)

    result = composite(fun, simplify=False)(value, configuration)

    assert observed == [configuration]
    assert observed[0] is configuration
    assert result.parents == (value,)
    assert result.op == ir.Composite(1, (ir.Log(),), ((0,),))


def test_composite_requires_one_new_rv_output():
    value = ir.RV(ir.Constant(1))

    with pytest.raises(ValueError, match="every output must be created"):
        composite(lambda x: x)(value)

    with pytest.raises(TypeError, match="exactly one newly created RV"):
        composite(lambda x: [ir.RV(ir.Exp(), x)])(value)  # type: ignore[arg-type]


def test_composite_rejects_a_random_intermediate_operation():
    mean = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))

    def fun(a: ir.RV, b: ir.RV) -> ir.RV:
        sample = ir.RV(ir.Normal(), a, b)
        return ir.RV(ir.Exp(), sample)

    with pytest.raises(ValueError, match="all but the last Op must be deterministic"):
        composite(fun)(mean, scale)


def test_composite_from_trace_respects_explicit_parent_order():
    first = ir.RV(ir.Constant(1))
    second = ir.RV(ir.Constant(2))
    output = ir.RV(ir.Sub(), first, second)

    op = composite_from_trace(output, [output], [second, first])

    assert op == ir.Composite(2, (ir.Sub(),), ((1, 0),))


def test_composite_from_trace_can_select_an_existing_output():
    first = ir.RV(ir.Constant(1))
    output = ir.RV(ir.Constant(2))

    op = composite_from_trace(output, [], [first, output], allow_existing_output=True)

    assert op == ir.Composite(2, (ir.Identity(),), ((1,),))
