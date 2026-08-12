from pangolin.interface import *
from pangolin import ir
from collections.abc import Callable

from pangolin.interface.vmapping import (
    convert_args,
    generated_nodes,
    vmap_dummy_args,
    AbstractOp,
    vmap_eval_flat,
    vmap,
)
import numpy as np


def test_1d_reduction():
    a = constant([1.1, 2.2, 3.3])
    b = constant([0, 1, 2])
    c = constant(5.5)
    d = vmap(lambda i: a[i] ** c)(b)
    assert d.op == ir.VMap(ir.Pow(), [0, None], 3)
    assert d.parents == (a, c)


def test_2d_reduction_ax0():
    a = constant([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    b = constant([0, 1])
    c = constant(5.5)
    d = vmap(lambda i: a[i, :] ** c)(b)
    assert d.op == ir.VMap(ir.VMap(ir.Pow(), [0, None], 3), [0, None], 2)
    assert d.parents == (a, c)


def test_2d_reduction_ax1():
    a = constant([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    b = constant([0, 1, 2])
    c = constant(5.5)
    d = vmap(lambda i: a[:, i] ** c)(b)
    assert d.op == ir.VMap(ir.VMap(ir.Pow(), [0, None], 2), [1, None], 3)
    assert d.parents == (a, c)


def test_output_reduction():
    a = constant([1.1, 2.2, 3.3])
    b = constant([0, 1, 2])
    d = vmap(lambda i: a[i])(b)
    print(f"{d=}")
    assert d == a

    # assert d.op == ir.VMap(ir.Pow(), [0, None], 3)
    # assert d.parents == (a, c)


def test_double_reduction():
    a = constant([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    b = vmap(vmap(lambda i, j: a[i, j], [None, 0], 3), [0, None], 2)(constant(range(2)), constant(range(3)))
    print(f"{b=}")
