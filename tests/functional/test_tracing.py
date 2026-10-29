import pytest

from pangolin import functional, ir
from pangolin.functional.tracing import (
    AbstractOp,
    flatten_rv_inputs,
    graph_scan,
    require_rv_leaves,
    trace_call,
)


def test_flatten_rv_inputs_returns_rvs_axes_and_positions():
    mapped = ir.RV(ir.Constant([1, 2]))
    unmapped = ir.RV(ir.Constant(3))

    leaves, rvs, axes, positions = flatten_rv_inputs(
        ([mapped, "exp"], unmapped), ([0, None], None), "test"
    )

    assert leaves == [mapped, "exp", unmapped]
    assert rvs == [mapped, unmapped]
    assert axes == [0, None]
    assert positions == [0, 2]


def test_public_api_contains_graph_scan():
    assert functional.graph_scan is graph_scan


def test_graph_scan_rebuilds_an_ir_graph_and_preserves_its_pytree():
    root = ir.RV(ir.Constant(1))
    leaf = ir.RV(ir.Exp(), root)

    def rebuild(node: ir.RV, parents: tuple[ir.RV, ...]) -> tuple[ir.RV, ir.Op]:
        return ir.RV(node.op, *parents, interface=node.interface), node.op

    replacements, emissions = graph_scan({"leaf": [leaf]}, rebuild)

    assert isinstance(replacements, dict)
    assert isinstance(replacements["leaf"], list)
    replacement = replacements["leaf"][0]
    assert isinstance(replacement, ir.RV)
    assert replacement is not leaf
    assert replacement.op == ir.Exp()
    assert replacement.parents[0] is not root
    assert emissions == {root: root.op, leaf: leaf.op}


def test_graph_scan_rejects_a_non_rv_carry():
    root = ir.RV(ir.Constant(1))

    with pytest.raises(TypeError, match="step must return an RV carry; got int"):
        graph_scan(root, lambda node, parents: (1, None))  # type: ignore


def test_trace_call_accepts_plain_ir_rvs_and_preserves_creation_order():
    root = ir.RV(ir.Constant(1))
    created: list[ir.RV] = []

    def fun(value: ir.RV):
        first = ir.RV(ir.Exp(), value)
        second = ir.RV(ir.Log(), first)
        created.extend([first, second])
        return {"second": second, "first": first}

    output, generated, boundary = trace_call(fun, root)

    assert generated == created
    assert boundary == [root]


def test_trace_call_forwards_keyword_named_fun():
    root = ir.RV(ir.Constant(1))

    def build(*, fun: ir.RV):
        return ir.RV(ir.Exp(), fun)

    output, generated, boundary = trace_call(build, fun=root)
    assert isinstance(output, ir.RV)
    assert output.parents == (root,)
    assert generated == [output]
    assert boundary == [root]


def test_trace_call_preserves_interfaces_selected_by_the_function():
    input_interface = ir.RVInterface()
    output_interface = ir.RVInterface()
    root = ir.RV(ir.Constant(1), interface=input_interface)

    def fun(value: ir.RV):
        return ir.RV(ir.Exp(), value, interface=output_interface)

    output, generated, boundary = trace_call(fun, root)

    assert type(output) is ir.RV
    assert output.interface is output_interface
    assert generated == [output]
    assert boundary == [root]


def test_trace_call_preserves_existing_outputs_as_boundary_nodes():
    interface = ir.RVInterface()
    root = ir.RV(ir.Constant(1), interface=interface)

    output, generated, boundary = trace_call(lambda value: [value], root)

    assert isinstance(output, list)
    assert output == [root]
    assert isinstance(output[0], ir.RV)
    assert output[0].interface is interface
    assert generated == []
    assert boundary == [root]


def test_require_rv_leaves_rejects_non_rv_leaves():
    with pytest.raises(TypeError, match="found NoneType"):
        require_rv_leaves({"bad": None}, "values")


def test_require_rv_leaves_returns_typed_leaves():
    root = ir.RV(ir.Constant(1))

    assert require_rv_leaves({"first": root, "second": [root]}, "values") == [root, root]


def test_abstract_op_has_a_fixed_shape_and_identity_equality():
    first = AbstractOp((2, 3))
    second = AbstractOp((2, 3))

    assert first.get_shape() == (2, 3)
    assert first != second
    assert repr(first) == "AbstractOp((2, 3))"
