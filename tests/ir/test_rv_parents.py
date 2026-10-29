from pangolin import interface as pi, ir
from pangolin.ir import RV


import pytest

from pangolin.interface import constant


def test_constructor_accepts_infix_rv_subclasses():
    class CustomRV(RV):
        pass

    parent = CustomRV(ir.Constant(1))
    child = RV(ir.Identity(), parent, interface=pi.default)

    assert child.parents == (parent,)


def test_abs_creates_abs_rv():
    x = constant(-2)
    y = abs(x)

    assert y.op == ir.Abs()
    assert y.parents == (x,)


def test_truth_value_testing_is_rejected():
    x = constant(1)

    with pytest.raises(TypeError, match="RVs do not have a truth value"):
        bool(x)
