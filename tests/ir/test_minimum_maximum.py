"""Scalar contracts for the minimum and maximum Ops."""

import pytest

from pangolin import ir


@pytest.mark.parametrize("op_class", [ir.Minimum, ir.Maximum])
def test_scalar_binary_op(op_class):
    op = op_class()
    assert op_class.__name__ in ir.__all__
    assert op.get_shape((), ()) == ()
    assert not op.random
    assert repr(op) == f"{op_class.__name__}()"
    assert str(op) == op_class.__name__.lower()
    assert op == op_class()
    assert hash(op) == hash(op_class())
    assert ir.Minimum() != ir.Maximum()
    for shapes in [(), ((),), ((), (), ())]:
        with pytest.raises(TypeError):
            op.get_shape(*shapes)
    for shapes in [((1,), ()), ((), (2,)), ((2,), (2,))]:
        with pytest.raises(ValueError, match="not all scalar"):
            op.get_shape(*shapes)


@pytest.mark.parametrize("op_class", [ir.Minimum, ir.Maximum])
def test_graph_construction_does_not_select_or_evaluate_a_parent(op_class):
    left = ir.RV(ir.Constant(float("-inf")))
    right = ir.RV(ir.Constant(float("inf")))
    result = ir.RV(op_class(), left, right)
    assert result.parents == (left, right)
    assert result.op == op_class()
    assert result.shape == ()
