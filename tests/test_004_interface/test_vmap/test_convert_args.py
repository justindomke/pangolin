from pangolin.interface import *
from pangolin import ir
from collections.abc import Callable

from pangolin.interface.vmap import (
    convert_args,
    generated_nodes,
    vmap_dummy_args,
    AbstractOp,
    vmap_eval,
    vmap
)
import numpy as np


class NewRV(OperatorRV):
    """
    New test RV class to make sure propagated correctly
    """
    def __repr__(self):
        return "New" + super().__repr__()[8:]  # len("Operator")=8


def test_convert_args_independent():
    x0 = normal(0, 1)
    y0 = normal(2, 3)
    x1, y1 = convert_args(NewRV, x0, y0)
    assert isinstance(x1, NewRV)
    assert isinstance(y1, NewRV)
    assert x1.parents == x0.parents
    assert y1.parents == y0.parents
    assert x1.op == ir.Normal()
    assert y1.op == ir.Normal()


def test_convert_args_dependent():
    x0 = normal(0, 1)
    y0 = normal(x0, 2)
    x1, y1 = convert_args(NewRV, x0, y0)
    assert isinstance(x1, NewRV)
    assert isinstance(y1, NewRV)
    assert x1.parents == x0.parents
    assert y1.parents == (x1, y0.parents[1])
    assert x1.op == ir.Normal()
    assert y1.op == ir.Normal()


def test_convert_args_closure():
    z = normal(0, 1) ** 2
    x0 = normal(0, z)
    y0 = normal(x0, z)
    x1, y1 = convert_args(NewRV, x0, y0)
    assert isinstance(x1, NewRV)
    assert isinstance(y1, NewRV)
    assert x1.parents == (x0.parents[0], z)
    assert y1.parents == (x1, z)
    assert x1.op == ir.Normal()
    assert y1.op == ir.Normal()

