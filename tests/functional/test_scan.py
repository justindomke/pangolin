import pytest

from pangolin import functional, ir
from pangolin.functional import scanning
from pangolin.functional.scanning import make_abstract_rvs, scan
from pangolin.functional.tracing import AbstractOp


def test_public_api_contains_scan():
    assert scanning.__all__ == ["scan"]
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
    assert functional.scan is scan


def test_scan_constructs_a_plain_ir_recurrence():
    start = ir.RV(ir.Constant(1.0))

    result = scan(lambda carry: ir.RV(ir.Exp(), carry), length=3)(start)

    assert type(result) is ir.RV
    assert result.parents == (start,)
    assert result.op == ir.Scan(
        ir.Exp(),
        3,
        (),
    )


def test_scan_names_the_returned_function():
    def exponential(carry: ir.RV) -> ir.RV:
        return ir.RV(ir.Exp(), carry)

    scanned_exponential = scan(exponential, length=3)

    assert getattr(scanned_exponential, "__name__") == "scan_exponential"
    assert getattr(scanned_exponential, "__qualname__") == "scan_exponential"


def test_scan_preserves_the_carry_interface():
    interface = ir.RVInterface()
    start = ir.RV(ir.Constant(1.0), interface=interface)
    observed: list[ir.RV] = []

    def fun(carry: ir.RV) -> ir.RV:
        observed.append(carry)
        return ir.RV(ir.Exp(), carry, interface=interface)

    result = scan(fun, length=3)(start)

    assert observed[0].interface is interface
    assert isinstance(observed[0].op, AbstractOp)
    assert type(result) is ir.RV
    assert result.interface is interface


def test_scan_does_not_require_argument_interfaces_to_match_the_carry():
    carry_interface = ir.RVInterface()
    values_interface = ir.RVInterface()
    start = ir.RV(ir.Constant(0), interface=carry_interface)
    values = ir.RV(ir.Constant([1, 2, 3]), interface=values_interface)

    result = scan(
        lambda carry, value: ir.RV(ir.Add(), carry, value, interface=carry_interface),
        length=3,
    )(start, values)

    assert type(result) is ir.RV
    assert result.interface is carry_interface
    assert result.parents == (start, values)


def test_abstract_rvs_preserve_each_argument_interface_and_remove_its_axis():
    first_interface = ir.RVInterface()
    second_interface = ir.RVInterface()
    first = ir.RV(ir.Constant([[1, 2], [3, 4], [5, 6]]), interface=first_interface)
    second = ir.RV(ir.Constant([7, 8]), interface=second_interface)

    abstract_rvs = make_abstract_rvs([first, second], [0, None], 3)

    assert abstract_rvs[0].interface is first_interface
    assert abstract_rvs[1].interface is second_interface
    assert [rv.shape for rv in abstract_rvs] == [(2,), (2,)]


def test_scan_rejects_a_step_output_with_a_different_interface():
    carry_interface = ir.RVInterface()
    output_interface = ir.RVInterface()
    start = ir.RV(ir.Constant(0), interface=carry_interface)

    with pytest.raises(ValueError, match="output interface must match"):
        scan(
            lambda carry: ir.RV(ir.Exp(), carry, interface=output_interface),
            length=3,
        )(start)


def test_scan_accepts_nested_argument_pytrees():
    start = ir.RV(ir.Constant(0))
    values = {
        "mapped": ir.RV(ir.Constant([1, 2, 3])),
        "fixed": ir.RV(ir.Constant(4)),
    }

    def fun(carry: ir.RV, tree: dict[str, ir.RV]) -> ir.RV:
        total = ir.RV(ir.Add(), carry, tree["mapped"])
        return ir.RV(ir.Add(), total, tree["fixed"])

    result = scan(fun, length=3, in_axes={"mapped": 0, "fixed": None})(start, values)

    assert result.parents == (start, values["fixed"], values["mapped"])
    assert result.op == ir.Scan(
        ir.Composite(3, (ir.Add(), ir.Add()), ((0, 2), (3, 1))),
        3,
        (None, 0),
    )


def test_scan_preserves_captures_after_explicit_arguments():
    start = ir.RV(ir.Constant(0))
    values = ir.RV(ir.Constant([1, 2, 3]))
    captured = ir.RV(ir.Constant(10))

    def fun(carry: ir.RV, value: ir.RV) -> ir.RV:
        total = ir.RV(ir.Add(), captured, carry)
        return ir.RV(ir.Add(), total, value)

    result = scan(fun, length=3)(start, values)

    assert result.parents == (start, values, captured)
    assert result.op == ir.Scan(
        ir.Composite(3, (ir.Add(), ir.Add()), ((2, 0), (3, 1))),
        3,
        (0, None),
    )


def test_scan_omits_unused_additional_arguments():
    start = ir.RV(ir.Constant(0))
    used = ir.RV(ir.Constant([1, 2, 3]))
    unused = ir.RV(ir.Constant([4, 5, 6]))

    result = scan(lambda carry, value, ignored: ir.RV(ir.Add(), carry, value), length=3)(
        start,
        used,
        unused,
    )

    assert result.parents == (start, used)
    assert result.op == ir.Scan(
        ir.Add(),
        3,
        (0,),
    )


def test_scan_requires_an_rv_initial_carry_without_coercion():
    with pytest.raises(TypeError, match="initial carry must be an RV; got int"):
        scan(lambda carry: carry, length=3)(1)  # type: ignore[arg-type]


def test_scan_passes_unmapped_non_rv_arguments_through_unchanged():
    start = ir.RV(ir.Constant(0))
    configuration = object()
    observed: list[object] = []

    def fun(carry: ir.RV, config: object) -> ir.RV:
        observed.append(config)
        return ir.RV(ir.Exp(), carry)

    result = scan(fun, length=3, in_axes=None)(start, configuration)

    assert observed == [configuration]
    assert observed[0] is configuration
    assert result.parents == (start,)


def test_scan_accepts_static_leaves_inside_argument_pytrees():
    start = ir.RV(ir.Constant(0))
    values = {
        "mapped": ir.RV(ir.Constant([1, 2, 3])),
        "operation": "add",
    }

    def fun(carry: ir.RV, tree: dict[str, object]) -> ir.RV:
        value = tree["mapped"]
        assert isinstance(value, ir.RV)
        op = ir.Add() if tree["operation"] == "add" else ir.Sub()
        return ir.RV(op, carry, value)

    result = scan(fun, length=3, in_axes={"mapped": 0, "operation": None})(start, values)

    assert result.parents == (start, values["mapped"])
    assert result.op == ir.Scan(
        ir.Add(),
        3,
        (0,),
    )


def test_scan_rejects_a_mapped_non_rv_argument():
    start = ir.RV(ir.Constant(0))

    with pytest.raises(TypeError, match="axis for non-RV argument must be None"):
        scan(lambda carry, value: carry, length=3)(start, 1)


def test_scan_validates_length_and_mapped_axis_size():
    start = ir.RV(ir.Constant(0))
    values = ir.RV(ir.Constant([1, 2]))

    with pytest.raises(ValueError, match="length must be positive"):
        scan(lambda carry: ir.RV(ir.Exp(), carry), length=0)(start)

    with pytest.raises(ValueError, match="mapped axis length 2, expected 3"):
        scan(lambda carry, value: ir.RV(ir.Add(), carry, value), length=3)(start, values)


def test_scan_requires_the_output_to_depend_on_the_carry():
    start = ir.RV(ir.Constant(0))
    values = ir.RV(ir.Constant([1, 2, 3]))

    with pytest.raises(ValueError, match="output must depend on its carry"):
        scan(lambda carry, value: ir.RV(ir.Exp(), value), length=3)(start, values)
