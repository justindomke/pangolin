from pangolin import interface as pi, ir
from pangolin.ir import RV


import numpy as np
import pytest

from pangolin.interface import constant
from pangolin.interface import exp
from pangolin.functional.scanning import scan


def test_scan_accepts_an_explicit_constant_carry():
    result = scan(lambda carry: exp(carry), length=3)(pi.default.coerce([1.0, 2.0]))

    assert result.parents[0].op == ir.Constant([1.0, 2.0])
    assert result.op == ir.Scan(
        ir.VMap(ir.Exp(), (0,), 2),
        3,
        (),
    )


@pytest.mark.parametrize(
    "argument, expected_type",
    [
        ([1, 2, 3], "int"),
        (np.array([1, 2, 3]), "ndarray"),
        (3, "int"),
    ],
)
def test_scan_rejects_non_rv_leaves_in_additional_arguments(argument, expected_type):
    fun = lambda carry, value: carry + value

    with pytest.raises(
        TypeError,
        match=rf"axis for non-RV argument must be None; got axis 0 for {expected_type}",
    ):
        scan(fun, length=3)(pi.default.coerce(constant(0)), argument)


def test_scan_rejects_nested_non_rv_mapped_leaf():
    argument = {"valid": constant([1, 2]), "invalid": [constant([3, 4]), 5]}

    with pytest.raises(TypeError, match="axis for non-RV argument must be None"):
        scan(lambda carry, values: carry, length=2)(pi.default.coerce(constant(0)), argument)


def test_scan_accepts_nested_pytrees_of_rvs():
    carry = constant(0)
    values = {
        "x": constant([1, 2, 3]),
        "scale": constant(4),
    }

    def fun(previous, tree):
        return previous + tree["x"] * tree["scale"]

    result = scan(fun, length=3, in_axes={"x": 0, "scale": None})(pi.default.coerce(carry), values)

    base_op = ir.Composite(3, (ir.Mul(), ir.Add()), ((2, 1), (0, 3)))
    assert result.op == ir.Scan(base_op, 3, (None, 0))
    assert result.parents == (carry, values["scale"], values["x"])


@pytest.mark.parametrize("in_axes", [(0, None), [0, None]])
def test_scan_accepts_lists_and_tuples_for_multiple_argument_axes(in_axes):
    carry = constant(0)
    values = constant([1, 2, 3])
    offset = constant(4)

    result = scan(
        lambda previous, value, fixed: previous + value + fixed,
        length=3,
        in_axes=in_axes,
    )(pi.default.coerce(carry), values, offset)

    assert result.op.in_axes == (0, None)
    assert result.parents == (carry, values, offset)


def test_scan_rejects_a_mapped_axis_with_the_wrong_length():
    values = constant([1, 2])

    with pytest.raises(ValueError, match="mapped axis length 2, expected 3"):
        scan(lambda previous, value: previous + value, length=3)(
            pi.default.coerce(constant(0)), values
        )


def test_scan_preserves_closure_captures_after_explicit_arguments():
    captured = constant(10)
    carry = constant(0)
    values = constant([1, 2, 3])

    def fun(previous, value):
        return captured + previous + value

    result = scan(fun, length=3)(pi.default.coerce(carry), values)

    assert result.parents == (carry, values, captured)
    assert result.op.base_op == ir.Composite(
        3,
        (ir.Add(), ir.Add()),
        ((2, 0), (3, 1)),
    )


def test_scan_calls_function_only_with_symbolic_rvs():
    observed = []

    def fun(carry, value):
        observed.append((carry, value))
        return carry + value

    scan(fun, length=3)(pi.default.coerce(constant(0)), constant([1, 2, 3]))

    assert observed
    assert all(isinstance(carry, RV) and isinstance(value, RV) for carry, value in observed)


def test_scan_requires_an_explicit_length():
    with pytest.raises(TypeError, match="length"):
        scan(lambda carry: exp(carry))  # type: ignore


def test_scan_removes_unused_additional_inputs():
    carry = constant(0)
    used = constant([1, 2, 3])
    unused = constant([4, 5, 6])

    result = scan(lambda previous, x, ignored: previous + x, length=3)(
        pi.default.coerce(carry),
        used,
        unused,
    )

    assert result.parents == (carry, used)
    assert result.op == ir.Scan(
        ir.Add(),
        3,
        (0,),
    )


def test_scan_rejects_a_function_whose_output_does_not_depend_on_carry():
    values = constant([1, 2, 3])

    with pytest.raises(ValueError, match="output must depend on its carry"):
        scan(lambda carry, value: exp(value), length=3)(pi.default.coerce(constant(0)), values)
