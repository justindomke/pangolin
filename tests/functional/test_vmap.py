import pytest

import pangolin
from pangolin import dag, functional, ir
from pangolin.functional import vmapping
from pangolin.functional.simplifying import permutation_view
from pangolin.functional.tracing import AbstractOp
from pangolin.functional.vmapping import make_abstract_rvs, vmap


def index_row(
    array: ir.RV,
    row: ir.RV,
    interface: ir.RVInterface = ir.NO_INTERFACE,
) -> ir.RV:
    """Select one row using only IR operations."""
    columns = ir.RV(ir.Constant(range(array.shape[1])), interface=interface)
    return ir.RV(
        ir.VMap(ir.Index(), (None, None, 0), array.shape[1]),
        array,
        row,
        columns,
        interface=interface,
    )


def contains_index(value: ir.RV) -> bool:
    """Return whether an Index occurs in the upstream graph."""

    def op_contains_index(op: ir.Op) -> bool:
        if isinstance(op, ir.Index):
            return True
        if isinstance(op, ir.VMap):
            return op_contains_index(op.base_op)
        return False

    return any(op_contains_index(node.op) for node in dag.upstream_nodes(value))


def test_public_api_contains_vmap():
    assert vmapping.__all__ == ["vmap"]
    assert functional.__all__ == [
        "broadcast_off",
        "broadcast_simple",
        "broadcast_rank",
        "broadcast_numpy",
        "composite",
        "index_orthogonal",
        "ir_code",
        "signature",
        "log_prob",
        "cdf",
        "quantile",
        "truncated",
        "grad",
        "vjp",
        "vmap",
        "scan",
        "loop",
        "vfor",
        "sfor",
        "bind_interface",
        "graph_scan",
        "Transform",
        "compose_transforms",
        "tforms",
        "transform_for",
        "unconstrain",
    ]
    assert functional.vmap is vmap


def test_pangolin_exports_functional_module():
    assert "functional" in pangolin.__all__
    assert pangolin.functional is functional


def test_vmap_constructs_a_plain_ir_graph():
    value = ir.RV(ir.Constant([1.0, 2.0, 3.0]))

    result = vmap(lambda x: ir.RV(ir.Exp(), x))(value)

    assert type(result) is ir.RV
    assert result.parents == (value,)
    assert result.op == ir.VMap(ir.Exp(), (0,), 3)


def test_vmap_names_the_returned_function():
    def exponential(value: ir.RV) -> ir.RV:
        return ir.RV(ir.Exp(), value)

    mapped_exponential = vmap(exponential)

    assert getattr(mapped_exponential, "__name__") == "vmap_exponential"
    assert getattr(mapped_exponential, "__qualname__") == "vmap_exponential"


def test_vmap_preserves_each_traced_node_interface():
    input_interface = ir.RVInterface()
    output_interface = ir.RVInterface()
    value = ir.RV(ir.Constant([1.0, 2.0, 3.0]), interface=input_interface)
    observed: list[ir.RV] = []

    def fun(x: ir.RV) -> ir.RV:
        observed.append(x)
        intermediate = ir.RV(ir.Exp(), x, interface=output_interface)
        return ir.RV(ir.Log(), intermediate, interface=output_interface)

    result = vmap(fun)(value)

    assert observed[0].interface is input_interface
    assert isinstance(observed[0].op, AbstractOp)
    assert type(result) is ir.RV
    assert result.interface is output_interface
    assert result.parents[0].interface is output_interface


def test_vmap_does_not_require_one_input_interface():
    mapped_interface = ir.RVInterface()
    fixed_interface = ir.RVInterface()
    mapped = ir.RV(ir.Constant([1, 2, 3]), interface=mapped_interface)
    fixed = ir.RV(ir.Constant(4), interface=fixed_interface)

    result = vmap(
        lambda x, y: ir.RV(ir.Add(), x, y),
        in_axes=(0, None),
    )(mapped, fixed)

    assert type(result) is ir.RV
    assert result.parents == (mapped, fixed)


def test_abstract_rvs_preserve_interfaces_and_remove_mapped_axes():
    first_interface = ir.RVInterface()
    second_interface = ir.RVInterface()
    first = ir.RV(ir.Constant([[1, 2], [3, 4], [5, 6]]), interface=first_interface)
    second = ir.RV(ir.Constant([7, 8]), interface=second_interface)

    abstract_rvs, size = make_abstract_rvs([first, second], [0, None], None)

    assert abstract_rvs[0].interface is first_interface
    assert abstract_rvs[1].interface is second_interface
    assert [rv.shape for rv in abstract_rvs] == [(2,), (2,)]
    assert size == 3


def test_vmap_accepts_nested_argument_and_output_pytrees():
    values = {
        "mapped": ir.RV(ir.Constant([1, 2, 3])),
        "fixed": ir.RV(ir.Constant(4)),
    }

    def fun(tree: dict[str, ir.RV]) -> dict[str, ir.RV]:
        total = ir.RV(ir.Add(), tree["mapped"], tree["fixed"])
        return {"total": total, "again": total}

    result = vmap(fun, in_axes={"mapped": 0, "fixed": None})(values)

    assert result["total"] is result["again"]
    assert result["total"].parents == (values["mapped"], values["fixed"])
    assert result["total"].op == ir.VMap(ir.Add(), (0, None), 3)


def test_vmap_passes_unmapped_non_rv_arguments_through_unchanged():
    value = ir.RV(ir.Constant([1, 2, 3]))
    configuration = object()
    observed: list[object] = []

    def fun(x: ir.RV, config: object) -> ir.RV:
        observed.append(config)
        return ir.RV(ir.Exp(), x)

    result = vmap(fun, in_axes=(0, None))(value, configuration)

    assert observed == [configuration]
    assert observed[0] is configuration
    assert result.parents == (value,)


def test_vmap_accepts_static_leaves_inside_argument_pytrees():
    values = {
        "mapped": ir.RV(ir.Constant([1, 2, 3])),
        "operation": "exp",
    }

    def fun(tree: dict[str, object]) -> ir.RV:
        value = tree["mapped"]
        assert isinstance(value, ir.RV)
        op = ir.Exp() if tree["operation"] == "exp" else ir.Log()
        return ir.RV(op, value)

    result = vmap(fun, in_axes={"mapped": 0, "operation": None})(values)

    assert result.op == ir.VMap(ir.Exp(), (0,), 3)
    assert result.parents == (values["mapped"],)


def test_vmap_rejects_a_mapped_non_rv_argument():
    with pytest.raises(TypeError, match="axis for non-RV argument must be None"):
        vmap(lambda value: ir.RV(ir.Constant(value)))(1)


def test_vmap_infers_nonzero_axis_and_validates_sizes():
    first = ir.RV(ir.Constant([[1, 2, 3], [4, 5, 6]]))

    result = vmap(lambda x: ir.RV(ir.Identity(), x), in_axes=1)(first)

    assert result.shape == (3, 2)
    assert result.op == ir.VMap(ir.Identity(), (1,), 3)

    expected_size = ir.RV(ir.Constant([1, 2, 3]))
    wrong_size = ir.RV(ir.Constant([1, 2]))
    with pytest.raises(ValueError, match="mapped axis size 2, expected 3"):
        vmap(lambda x, y: ir.RV(ir.Add(), x, y))(expected_size, wrong_size)


def test_vmap_requires_axis_size_without_a_mapped_rv():
    fixed = ir.RV(ir.Constant(1))

    with pytest.raises(ValueError, match="axis_size is required"):
        vmap(lambda x: ir.RV(ir.Exp(), x), in_axes=None)(fixed)

    result = vmap(
        lambda x: ir.RV(ir.Exp(), x),
        in_axes=None,
        axis_size=4,
    )(fixed)
    assert result.op == ir.VMap(ir.Exp(), (None,), 4)


def test_vmap_supports_a_zero_argument_function_with_axis_size():
    result = vmap(lambda: ir.RV(ir.Constant(2)), axis_size=3)()

    assert result.parents == ()
    assert result.op == ir.VMap(ir.Constant(2), (), 3)


def test_vmap_preserves_captures_and_omits_unused_inputs():
    used = ir.RV(ir.Constant([1, 2, 3]))
    unused = ir.RV(ir.Constant([4, 5, 6]))
    captured = ir.RV(ir.Constant(10))

    def fun(x: ir.RV, ignored: ir.RV) -> ir.RV:
        return ir.RV(ir.Add(), captured, x)

    result = vmap(fun)(used, unused)

    assert result.parents == (captured, used)
    assert result.op == ir.VMap(ir.Add(), (None, 0), 3)


def test_vmap_requires_new_rv_outputs():
    value = ir.RV(ir.Constant([1, 2, 3]))

    with pytest.raises(ValueError, match="every output must be created"):
        vmap(lambda x: x)(value)

    with pytest.raises(TypeError, match="trace_call output must be a PyTree"):
        vmap(lambda x: [ir.RV(ir.Exp(), x), None])(value)  # type: ignore[list-item]


def test_vmap_calls_the_function_once_with_an_abstract_input():
    value = ir.RV(ir.Constant([1, 2, 3]))
    calls: list[ir.RV] = []

    def fun(x: ir.RV) -> ir.RV:
        calls.append(x)
        return ir.RV(ir.Exp(), x)

    vmap(fun)(value)

    assert len(calls) == 1
    assert isinstance(calls[0].op, AbstractOp)


def test_vmap_eliminates_an_identity_index_view():
    matrix = ir.RV(ir.Constant([[1, 2, 3], [4, 5, 6]]))
    rows = ir.RV(ir.Constant(range(2)))

    result = vmap(lambda row: index_row(matrix, row))(rows)

    assert result is matrix


def test_vmap_keeps_a_view_when_reusing_its_base_would_change_interface():
    matrix_interface = ir.RVInterface()
    result_interface = ir.RVInterface()
    matrix = ir.RV(ir.Constant([[1, 2, 3], [4, 5, 6]]), interface=matrix_interface)
    rows = ir.RV(ir.Constant(range(2)))

    result = vmap(lambda row: index_row(matrix, row, result_interface))(rows)

    assert type(result) is ir.RV
    assert result.interface is result_interface
    assert result is not matrix
    assert contains_index(result)
    assert permutation_view(result) == (matrix, (0, 1))


def test_vmap_can_disable_view_simplification():
    matrix = ir.RV(ir.Constant([[1, 2, 3], [4, 5, 6]]))
    rows = ir.RV(ir.Constant(range(2)))

    result = vmap(lambda row: index_row(matrix, row), simplify=False)(rows)

    assert result is not matrix
    assert contains_index(result)
    assert permutation_view(result) == (matrix, (0, 1))
