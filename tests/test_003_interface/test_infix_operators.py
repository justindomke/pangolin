from pangolin.interface.interface import OperatorRV
from pangolin import ir

# from pangolin.ir.op import SetAutoRV
from pangolin.ir import RV, Constant
from pangolin.interface import makerv
import numpy as np


def test_add():
    x = makerv(2)
    y = makerv(3)
    z = x + y
    assert z.op == ir.Add()
    assert z.parents == (x, y)


def test_add_implicit():
    x = makerv(2)
    z = x + 3
    assert z.op == ir.Add()
    assert z.parents[0] == x
    assert z.parents[1].op == Constant(3)


def test_sub():
    x = makerv(2)
    y = 3
    z = x - y
    assert z.op == ir.Sub()
    assert z.parents[0] == x
    assert z.parents[1].op == Constant(3)


def test_matmul_rv():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    y = makerv([1, 2, 3])
    z = x @ y
    assert z.op == ir.MatMul()


def test_matmul_list():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    y = [1, 2, 3]
    z = x @ y
    assert z.op == ir.MatMul()


def test_matmul_numpy():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2, 3])
    z = x @ y
    assert z.op == ir.MatMul()


def test_rmatmul_list():
    x = [[1, 2, 3], [4, 5, 6]]
    y = makerv([1, 2, 3])
    z = x @ y
    assert z.op == ir.MatMul()


def test_rmatmul_numpy():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = makerv([1, 2, 3])
    z = x @ y
    assert z.op == ir.MatMul()


def test_pow():
    z = makerv(2.0) ** 3.0
    assert z.op == ir.Pow()
    assert z.parents[0].op == Constant(2.0)
    assert z.parents[1].op == Constant(3.0)


def test_rpow():
    z = 2.0 ** makerv(3.0)
    assert z.op == ir.Pow()
    assert z.parents[0].op == Constant(2.0)
    assert z.parents[1].op == Constant(3.0)
