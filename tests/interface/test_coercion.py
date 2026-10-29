from pangolin import interface as pi, ir
from pangolin.ir import RV


from typing import assert_type

import numpy as np
import pytest

from pangolin.interface import constant
from pangolin.interface import normal
from pangolin.interface import overload as override


def test_coerce_to_rv_preserves_interface_rv_identity():
    rv = constant(1)

    assert_type(rv, RV)
    coerced = pi.default.coerce(rv)

    assert_type(coerced, RV)
    assert coerced is rv


def test_coerce_to_rv_creates_constant():
    rv = pi.default.coerce([1, 2])

    assert_type(rv, RV)
    assert isinstance(rv, RV)
    assert isinstance(rv.op, ir.Constant)
    np.testing.assert_array_equal(rv.op.value, [1, 2])


def test_rv_with_coerced_parents_preserves_rvs_and_coerces_values():
    left = constant(1)
    result = pi.apply_op(pi.default, ir.Add(), left, 2)

    assert_type(result, RV)
    assert result.parents[0] is left
    assert isinstance(result.parents[1].op, ir.Constant)
    assert result.shape == ()


def test_rv_with_coerced_parents_validates_parent_shapes():
    with pytest.raises(ValueError):
        pi.apply_op(pi.default, ir.Matmul(), 1, 2)


def test_scalar_function_can_construct_nested_vmaps():
    with override(broadcasting="rank"):
        rv = normal(np.zeros((2, 3)), 1)

    assert_type(rv, RV)
    assert isinstance(rv.op, ir.VMap)
    assert isinstance(rv.op.base_op, ir.VMap)
    assert isinstance(rv.op.base_op.base_op, ir.Normal)
