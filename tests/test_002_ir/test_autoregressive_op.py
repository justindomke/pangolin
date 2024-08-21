import pytest
from pangolin.ir.autoregressive import Autoregressive
from pangolin.ir.composite import Composite

# from pangolin.ir import *
from pangolin import ir
import numpy as np


def test_exponential():
    # op = Autoregressive(ir.Exponential(), 5, in_axes=(), where_self=0)
    op = Autoregressive(ir.Exponential(), 5, ())
    assert op.get_shape(()) == (5,)


def test_multi_normal_fixed_cov():
    ndims = 3
    length = 5
    op = Autoregressive(ir.MultiNormal(), length, (None,))
    assert op.get_shape((ndims,), (ndims, ndims)) == (length, ndims)


def test_multi_normal_varying_cov():
    ndims = 3
    length = 5
    op = Autoregressive(ir.MultiNormal(), length, (0,))
    assert op.get_shape((ndims,), (length, ndims, ndims)) == (length, ndims)


def test_multi_normal_varying_cov_dim1():
    ndims = 3
    length = 5
    op = Autoregressive(ir.MultiNormal(), length, (1,))
    assert op.get_shape((ndims,), (ndims, length, ndims)) == (length, ndims)


def test_multi_normal_varying_cov_dim2():
    ndims = 3
    length = 5
    op = Autoregressive(ir.MultiNormal(), length, (2,))
    assert op.get_shape((ndims,), (ndims, ndims, length)) == (length, ndims)


def test_matrix_mul():
    ndims = 3
    length = 5
    op = Autoregressive(ir.MatMul(), length, (None,), 1)
    assert op.get_shape((ndims,), (ndims, ndims)) == (length,ndims)

def test_matrix_mul_left():
    ndims = 3
    length = 5
    op = Autoregressive(ir.MatMul(), length, (None,), 0)
    assert op.get_shape((ndims,), (ndims, ndims)) == (length,ndims)

def test_vmap_autoregressive():
    dims = 3
    length = 5
    base_op = ir.VMap(ir.Mul(),(0,0))
    op = Autoregressive(base_op, length, (None,), 1)
    assert op.get_shape((dims,),(dims,)) == (length,dims)

def test_vmap_autoregressive_mapped0():
    dims = 3
    length = 5
    base_op = ir.VMap(ir.Mul(),(0,0))
    op = Autoregressive(base_op, length, (0,), 1)
    assert op.get_shape((dims,),(length,dims)) == (length,dims)

def test_vmap_autoregressive_mapped1():
    dims = 3
    length = 5
    base_op = ir.VMap(ir.Mul(),(0,0))
    op = Autoregressive(base_op, length, (1,), 1)
    assert op.get_shape((dims,),(dims,length)) == (length,dims)