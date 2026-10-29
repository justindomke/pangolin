from pangolin import interface as pi
from pangolin.ir import RV


import pytest

from pangolin.interface import constant
from pangolin.functional.tracing import AbstractOp


def test_abstract_op_requires_zero_parents():
    with pytest.raises(TypeError):
        RV(AbstractOp(), constant(1), interface=pi.default)


@pytest.mark.parametrize("shape", [[2], (2.0,), (True,)])
def test_abstract_op_requires_tuple_of_integers(shape):
    with pytest.raises(TypeError, match="shape must be a tuple of integers"):
        AbstractOp(shape)


@pytest.mark.parametrize("shape", [(0,), (-1,)])
def test_abstract_op_requires_positive_dimensions(shape):
    with pytest.raises(ValueError, match="dimensions must be positive"):
        AbstractOp(shape)
