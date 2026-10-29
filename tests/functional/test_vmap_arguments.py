from pangolin import ir
from pangolin.ir import RV


import numpy as np
import pytest

from pangolin.interface import constant
from pangolin.interface import exp
from pangolin.functional.vmapping import vmap


@pytest.mark.parametrize(
    "argument, expected_type",
    [
        ([1, 2, 3], "int"),
        (np.array([1, 2, 3]), "ndarray"),
        (3, "int"),
    ],
)
def test_vmap_rejects_non_rv_leaves(argument, expected_type):
    with pytest.raises(
        TypeError,
        match=rf"axis for non-RV argument must be None; got axis 0 for {expected_type}",
    ):
        vmap(exp)(argument)


def test_vmap_rejects_nested_non_rv_mapped_leaf():
    argument = {"valid": constant([1, 2]), "invalid": [constant([3, 4]), 5]}

    with pytest.raises(TypeError, match="axis for non-RV argument must be None"):
        vmap(lambda values: values["valid"])(argument)


def test_vmap_accepts_nested_pytrees_of_rvs():
    values = {
        "x": constant([1, 2, 3]),
        "scale": constant(4),
    }

    def fun(tree):
        return tree["x"] * tree["scale"]

    result = vmap(fun, {"x": 0, "scale": None})(values)

    assert result.op == ir.VMap(ir.Mul(), (0, None), 3)
    assert result.parents == (values["x"], values["scale"])


def test_vmap_accepts_lists_when_their_leaves_are_rvs():
    left = constant([1, 2, 3])
    right = constant([4, 5, 6])

    result = vmap(lambda values: values[0] + values[1], [0, 0])([left, right])

    assert result.op == ir.VMap(ir.Add(), (0, 0), 3)
    assert result.parents == (left, right)


def test_vmap_calls_function_only_with_symbolic_rvs():
    observed = []

    def fun(x):
        observed.append(x)
        return exp(x)

    x = constant([1, 2, 3])
    result = vmap(fun)(x)

    assert observed
    assert all(isinstance(value, RV) for value in observed)
    assert result.op == ir.VMap(ir.Exp(), (0,), 3)
    assert result.parents == (x,)
