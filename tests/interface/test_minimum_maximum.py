"""Minimum and maximum use the same configurable broadcasting as other scalar functions."""

from typing import assert_type

import pytest

from pangolin import interface as pi, ir


@pytest.mark.parametrize("name,op_class", [("minimum", ir.Minimum), ("maximum", ir.Maximum)])
def test_named_parents_exports_and_ownership(name, op_class):
    assert name in pi.__all__
    owner, foreign = pi.Interface(), pi.Interface()
    a, b = foreign.constant(2), foreign.constant(3)
    for interface, fun in [(owner, getattr(owner, name)), (pi.default, getattr(pi, name))]:
        result = fun(b=b, a=a)
        assert result.op == op_class()
        assert result.parents == (a, b)
        assert result.interface is interface
    assert_type(pi.minimum(2, 3), ir.RV)
    assert_type(pi.maximum(2, 3), ir.RV)


@pytest.mark.parametrize("name", ["minimum", "maximum"])
def test_broadcasting_modes_and_call_time_overload(name):
    strict = pi.Interface(broadcasting="off")
    fun = getattr(strict, name)
    assert fun(1, 2).shape == ()
    with pytest.raises(ValueError):
        fun([1, 2], 0)
    with strict.overload(broadcasting="simple"):
        assert fun([1, 2], 0).shape == (2,)
        assert fun([1, 2], [3, 4]).shape == (2,)
        with pytest.raises(ValueError):
            fun([[1, 2], [3, 4]], [1, 2])
    with strict.overload(broadcasting="rank"):
        assert fun([[1, 2], [3, 4]], [1, 2]).shape == (2, 2)
        with pytest.raises(ValueError):
            fun([[1], [2]], [1, 2, 3])
    with strict.overload(broadcasting="numpy"):
        assert fun([[1], [2]], [1, 2, 3]).shape == (2, 3)
    with pytest.raises(ValueError):
        fun([1, 2], 0)
